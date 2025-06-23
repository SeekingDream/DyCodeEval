from typing import Any, Dict, Optional, List, Iterable, Tuple
import abc
import torch
import numpy as np
import collections
from utils.fsc_datasets import PromptedClassificationDataset
from rewards.text_classification_reward import PromptedClassificationReward
from .test_time_bn import BatchNormCalibrate


class BaseTrainer(abc.ABC):
    """
    The base trainer class.

    Attributes:
        obj_func: the callable function handle for model interfacing.
        logger: an optional logger object.
        bn_calibrator: a batch norm calibration object. Only used in
            testing (not training or validation).
    """

    def __init__(
        self,
        obj_func: PromptedClassificationReward,
        prompt_dataset: PromptedClassificationDataset,
        logger: Optional[Any] = None,
        use_bn_calibrator: bool = False,
        n_samples_bn_calibrator: int = 128,
    ):
        self.obj_func = obj_func
        self.logger = logger
        self.prompt_dataset = prompt_dataset

        self.bn_calibrator = BatchNormCalibrate() if use_bn_calibrator else None
        self.n_samples_bn_calibrator = n_samples_bn_calibrator

    @abc.abstractmethod
    def train(self, train_data: Iterable[Any]):
        raise NotImplementedError()

    def validate(self, val_dataset: Iterable[Any], best_str_list: List[str]) -> str:
        t_dataset = val_dataset
        if self.logger is not None:
            self.logger.info("total val dataset length: %s", len(t_dataset))
        val_acc_list = []

        for prompt in best_str_list:
            n_correct = 0

            for batch_idx in range(0, len(t_dataset) // self.eval_batch_size + 1):
                idx = np.arange(
                    batch_idx * self.eval_batch_size,
                    (batch_idx + 1) * self.eval_batch_size,
                )
                idx = [_idx for _idx in idx if _idx < len(t_dataset)]

                if len(idx) == 0:
                    break

                t_data = [t_dataset[int(i)] for i in idx]
                (
                    t_premise_texts,
                    t_hypothesis,
                    t_class_labels,
                ) = self.prompt_dataset.get_data(t_data)

                torch.cuda.empty_cache()
                _, _, batch_acc = self.obj_func.forward(
                    t_premise_texts,
                    t_hypothesis,
                    t_class_labels,
                    prompt,
                    True,
                    "infer",
                    verbose=True,
                )
                n_correct += batch_acc * len(idx)
                torch.cuda.empty_cache()

            if self.logger is not None:
                self.logger.info("prompt: %s", prompt)
                self.logger.info("final val acc: %s", (n_correct / len(t_dataset)))
            val_acc_list.append(float(n_correct / len(t_dataset)))
        # best_prompt = best_str_list[np.argmax(val_acc_list)]
        max_acc = np.max(val_acc_list)
        indices = np.argwhere(val_acc_list == max_acc)
        last_index = indices[-1][0]
        best_prompt = best_str_list[last_index]
        if self.logger is not None:
            self.logger.info("val acc list: %s", val_acc_list)
            self.logger.info("best prompt: %s", best_prompt)
            self.logger.info("best prompt acc: %s", np.max(val_acc_list))

        return best_prompt

    def test(
        self,
        test_dataset,
        best_prompt,
        bn_calibrate_if_available: bool = True,
        return_logits: bool = False,
    ) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
        t_dataset = test_dataset
        if self.logger is not None:
            self.logger.info("total test dataset length: %s", len(t_dataset))
        n_correct = 0

        if self.bn_calibrator is not None and bn_calibrate_if_available:
            # select some samples for calibration
            idx_calibrate = np.random.choice(
                len(test_dataset),
                min(len(test_dataset), self.n_samples_bn_calibrator),
                replace=False,
            )

            calibrate_data = [t_dataset[int(i)] for i in idx_calibrate]
            (
                t_premise_texts,
                t_hypothesis,
                _,
            ) = self.prompt_dataset.get_data(calibrate_data)

            # Initialize the bn calibrator
            self.bn_calibrator.train()
            # Get the logits
            calibrate_logits = self.obj_func.forward(
                t_premise_texts,
                t_hypothesis,
                [0] * len(t_premise_texts),  # dummy class labels
                best_prompt,
                to_tensor=True,
                mode="infer",
                accumulate_class=True,
            )[-1]
            # Run the prediction logits only through the BN calibrator to obtain
            # running statistics.
            self.bn_calibrator(calibrate_logits[0], flush=True)
            self.bn_calibrator.eval()
            self.obj_func.bn_calibrator = self.bn_calibrator
        else:
            calibrate_logits = None

        all_logits: List[torch.Tensor] = []
        all_labels: List[int] = []
        for batch_idx in range(0, len(t_dataset) // self.eval_batch_size + 1):
            idx = np.arange(
                batch_idx * self.eval_batch_size, (batch_idx + 1) * self.eval_batch_size
            )
            idx = [_idx for _idx in idx if _idx < len(t_dataset)]

            if len(idx) == 0:
                break

            t_data = [t_dataset[int(i)] for i in idx]
            (
                t_premise_texts,
                t_hypothesis,
                t_class_labels,
            ) = self.prompt_dataset.get_data(t_data)

            torch.cuda.empty_cache()
            (
                _,
                _,
                batch_acc,
                _,
                _,
                _,
                class_logits,
            ) = self.obj_func.forward(
                t_premise_texts,
                t_hypothesis,
                t_class_labels,
                best_prompt,
                True,
                "infer",
                verbose=True,
                accumulate_class=True,
            )
            n_correct += batch_acc * len(idx)
            torch.cuda.empty_cache()
            if return_logits:
                all_logits.append(class_logits[0])
                all_labels += t_class_labels
        if self.logger is not None:
            self.logger.info("prompt: %s", best_prompt)
            self.logger.info(n_correct)
            self.logger.info("final test acc: %s", (n_correct / len(t_dataset)))
        if return_logits:
            return n_correct / len(t_dataset), {
                "output_logits": torch.cat(all_logits),
                "calibrate_logits": calibrate_logits,
                "labels": all_labels,
            }
        return n_correct / len(t_dataset), None

    def manual(
        self,
        test_dataset: Iterable[Any],
        bn_calibrate_if_available: bool = True,
        return_logits: bool = False,
    ) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
        t_dataset = test_dataset
        for i in range(self.n_classes):
            test_I = [x for x in t_dataset if x["label"] == i]
            if self.logger is not None:
                self.logger.info(
                    "total test dataset length: %s for class %s", len(test_I), i
                )
        if self.logger is not None:
            self.logger.info("total test dataset length: %s", len(t_dataset))
        n_correct = 0
        sum_ece = 0
        sum_entropy = 0
        class_correct = collections.Counter((i, 0) for i in range(self.n_classes))

        if self.bn_calibrator is not None and bn_calibrate_if_available:
            # select some samples for calibration
            idx_calibrate = np.random.choice(
                len(test_dataset),
                min(len(test_dataset), self.n_samples_bn_calibrator),
                replace=False,
            )

            calibrate_data = [t_dataset[int(i)] for i in idx_calibrate]
            (
                t_premise_texts,
                t_hypothesis,
                _,
            ) = self.prompt_dataset.get_data(calibrate_data)

            # Initialize the bn calibrator
            self.bn_calibrator.train()
            # Get the logits
            calibrate_logits = self.obj_func.forward(
                t_premise_texts,
                t_hypothesis,
                [0] * len(t_premise_texts),  # dummy class labels
                "",
                to_tensor=True,
                mode="infer",
                accumulate_class=True,
            )[-1]
            # Run the prediction logits only through the BN calibrator to obtain
            # running statistics.
            self.bn_calibrator(calibrate_logits[0], flush=True)
            self.bn_calibrator.eval()
            self.obj_func.bn_calibrator = self.bn_calibrator
        else:
            calibrate_logits = None

        all_logits: List[torch.Tensor] = []
        all_labels: List[int] = []
        for batch_idx in range(0, len(t_dataset) // self.eval_batch_size + 1):
            idx = np.arange(
                batch_idx * self.eval_batch_size, (batch_idx + 1) * self.eval_batch_size
            )
            idx = [_idx for _idx in idx if _idx < len(t_dataset)]

            if len(idx) == 0:
                break

            t_data = [t_dataset[int(i)] for i in idx]
            (
                t_premise_texts,
                t_hypothesis,
                t_class_labels,
            ) = self.prompt_dataset.get_data(t_data)

            torch.cuda.empty_cache()
            (
                _,
                _,
                batch_acc,
                count_class,
                batch_ece,
                batch_entropy,
                class_logits,
            ) = self.obj_func.forward(
                t_premise_texts,
                t_hypothesis,
                t_class_labels,
                "",
                True,
                "infer",
                verbose=True,
                accumulate_class=True,
            )
            n_correct += batch_acc * len(idx)
            sum_ece += batch_ece * len(idx)
            sum_entropy += batch_entropy * len(idx)
            class_correct += count_class[0]
            if return_logits:
                all_logits.append(class_logits[0])
                all_labels += t_class_labels
            # print(count_class)
            torch.cuda.empty_cache()
        # print(class_correct)
        if self.logger is not None:
            self.logger.info(
                "manual prompt test acc: %s", (float(n_correct) / len(t_dataset))
            )
            self.logger.info("count class: %s", class_correct)
            self.logger.info(
                "manual prompt test ece percent: %s",
                (float(sum_ece) / len(t_dataset) * 100),
            )
            self.logger.info(
                "manual prompt test entropy: %s", (float(sum_entropy) / len(t_dataset))
            )
        if return_logits:
            return float(n_correct) / len(t_dataset), {
                "output_logits": torch.cat(all_logits),
                "calibrate_logits": calibrate_logits,
                "labels": all_labels,
            }
        return float(n_correct) / len(t_dataset), None
