import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from utils.meters import ProgressMeter, AverageMeter
import torch.distributed as dist
import pandas as pd
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from algs.test_time_bn import BatchNormCalibrate

SUPPORTED_LEFT_TO_RIGHT_LMS = [
    "distilgpt2",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "t5-large",
    "flan-t5-large",
    "t5-base",
    "google/flan-t5-large",
]
SUPPORTED_MASK_LMS = [
    "distilroberta-base",
    "roberta-base",
    "roberta-large",
    "xlm-roberta-large",
]


def validate_text_classification_multiple_seeds(
    data_loaders: List[torch.utils.data.DataLoader],
    reward_module: "PromptedClassificationReward",
    output_tokens,
    gpu_id: Optional[int] = None,
    distributed_val: bool = False,
    logger=None,
    verbose: bool = False,
    print_freq: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A helper function that evaluates on multiple data_loaders across different seed
    """
    all_tops1 = {}
    all_rewards = {}
    for i, data_loader in enumerate(data_loaders):
        top1s, rewards = validate_text_classification(
            data_loader,
            reward_module,
            output_tokens,
            gpu_id,
            distributed_val,
            logger,
            verbose,
            print_freq,
        )
        top1s = pd.Series(top1s)
        rewards = pd.Series(rewards)
        all_tops1[i] = top1s
        all_rewards[i] = rewards
    all_top1s = pd.concat(all_top1s, axis=1)
    all_rewards = pd.concat(all_rewards, axis=1)
    return all_top1s, all_rewards


def validate_text_classification(
    data_loader: torch.utils.data.DataLoader,
    reward_module: "PromptedClassificationReward",
    output_tokens,
    gpu_id: Optional[int] = None,
    distributed_val: bool = False,
    logger=None,
    verbose: bool = False,
    print_freq: int = 10,
) -> Tuple[List[float], List[float]]:
    n_prompts = len(output_tokens) if isinstance(output_tokens, list) else 1
    batch_time = AverageMeter("Time", ":4.2f")
    rewards = [AverageMeter(f"Reward_Prompt{_}", ":4.1f") for _ in range(n_prompts)]
    top1 = [AverageMeter(f"Acc_Prompt{_}", ":4.2f") for _ in range(n_prompts)]

    progress = ProgressMeter(
        len(data_loader),
        [batch_time] + top1 + rewards,
        prefix="Validation: ",
        logger=logger,
    )
    if gpu_id is not None:
        reward_module = reward_module.to(gpu_id)

    # evaluation
    end = time.time()
    for batch_idx, data_item in enumerate(data_loader):
        sentences, targets = data_item["source_texts"], data_item["class_labels"]
        batch_size = len(sentences)
        rewards_tensor, accs_tensor, rewards_log = reward_module(
            source_texts=sentences,
            class_labels=targets,
            output_tokens=output_tokens,
            to_tensor=True,
            mode="infer",
            verbose=verbose,
        )

        if distributed_val:
            assert gpu_id is not None
            accs_shape = accs_tensor.shape[0]
            corr_ = accs_tensor * batch_size
            reward_ = rewards_tensor * batch_size
            stats = torch.cat([corr_, reward_, torch.tensor(batch_size).to(corr_)]).to(
                gpu_id
            )
            dist.barrier()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            corr_ = stats[:accs_shape]
            reward_ = stats[accs_shape : accs_shape * 2]
            batch_size = stats[-1]
            accs_tensor = corr_ / batch_size
            rewards_tensor = reward_ / batch_size

        for i, reward in enumerate(rewards_tensor):
            top1[i].update(accs_tensor[i], batch_size)
            rewards[i].update(reward, batch_size)
            batch_time.update(time.time() - end)
        if print_freq > 0 and batch_idx % print_freq:
            progress.display(batch_idx)
    # average the time and acc and return
    top1s = [float(m.avg) for i, m in enumerate(top1)]
    rewards = [float(m.avg) for m in rewards]
    if logger is not None:
        logger.info(f" * Acc@1 {top1s}.")
    return top1s, rewards


def _compute_reward(
    output: torch.Tensor,
    target: List[int],
    verbalizer_ids,
    reward_type: str,
    correct_coeff: float,
    incorrect_coeff: float,
    bn_calibrator: Optional[BatchNormCalibrate] = None,
):
    """
    Compute the accuracy over output (the logits from language models) and target
      the tensor of ground-truths and the GAP reward proposed in RLPrompt paper.
    """
    batch_size = output.shape[0]
    class_logits = output[:, verbalizer_ids]

    # This standardize the class logits
    if bn_calibrator is not None:
        bn_calibrator.eval()
        class_logits = bn_calibrator(class_logits)

    class_probs = torch.softmax(class_logits, -1)
    entropy = -(class_probs * torch.log(class_probs)).sum()
    predicted_label = torch.argmax(class_probs, -1)
    predicted_label_prob = class_probs[range(batch_size), predicted_label]
    # Get label and maximum not-label probabilities
    label_probs = class_probs[range(batch_size), target]
    not_label_probs = torch.where(
        class_probs == label_probs.unsqueeze(1),
        torch.Tensor([-1]).to(output.device),
        class_probs,
    )
    max_not_label_probs, _ = torch.max(not_label_probs, -1)
    # Compute piecewise gap reward
    gap = label_probs - max_not_label_probs
    correct = (gap > 0).long()
    if reward_type == "gap":
        gap_rewards = gap * (correct_coeff * correct + incorrect_coeff * (1 - correct))
        reward = gap_rewards.mean().detach()
    elif reward_type == "cross_entropy":
        criterion = CrossEntropyLoss()
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        reward = -1 * criterion(class_probs, target.to(class_probs.device))
    elif reward_type == "entropy":
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        reward = -1 * entropy
    # Log quantities such as accuracy and class-wise reward
    acc = correct.float().mean()
    # compute the confidence of the prediction
    conf = predicted_label_prob.mean()
    correct_predictions = Counter(
        target[i].item() for i in range(batch_size) if correct[i].item() == 1
    )
    return reward, acc, correct_predictions, conf, entropy, class_logits


def _compute_entropy(
    output: torch.Tensor,
    target: List[int],
    verbalizer_ids,
    reward_type: str,
    correct_coeff: float,
    incorrect_coeff: float,
):
    """
    Compute the accuracy over output (the logits from language models) and target
      the tensor of ground-truths and the GAP reward proposed in RLPrompt paper.
    """
    class_probs = torch.softmax(output[:, verbalizer_ids], -1)
    print("class_prob", class_probs)
    # Get label and maximum not-label probabilities
    entropy = -(class_probs * torch.log(class_probs)).sum()
    return entropy


def _compute_probs(
    output: torch.Tensor,
    target: List[int],
    verbalizer_ids,
    reward_type: str,
    correct_coeff: float,
    incorrect_coeff: float,
):
    """
    Compute the accuracy over output (the logits from language models) and target
      the tensor of ground-truths and the GAP reward proposed in RLPrompt paper.
    """
    class_probs = torch.softmax(output[:, verbalizer_ids], -1)
    return class_probs


class PromptedClassificationReward:
    def __init__(
        self,
        args,
        task_lm: str,
        is_mask_lm: Optional[bool],
        num_classes: int,
        verbalizers: List[str],
        reward_type: str = "entropy",
        compute_zscore: bool = True,
        incorrect_coeff: float = 180.0,  # lambda_1 in paper
        correct_coeff: float = 200.0,  # lambda_2 in paper
        use_bn_calibration: bool = False,
        bn_calibrator: Optional[BatchNormCalibrate] = None,
        template: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ):
        """
        Few shot text classification reward (adapted from RLPrompt repository)
        Args:
          task_lm: the string specifying the language model type of the task LM
          is_mask_lm: bool. Whether the LM is masked, or left-to-right.
          compute_zscore: bool. Whether do reward normalization by normalizing the
            mean and standard deviation across the batch.
          incorrect_coeff, correct_coeff:
          num_classes: number of classes in the labels
          verbalizers: a list of verbalizers (for e.g., for sentiment classification)
          reward_type: the type of the reward.
            "gap" -- use the one proposed in RLPrompt
            "ll" -- use the usual cross entropy loss
          template: the template to organize the queries and prompts.
            default one is [Input][Prompt][MASK].
            default template is adopted when it is not specified.
          bn_calibrator: an optional batch norm calibrator. When provided,
            in inference mode the logits will be first normalised by it first. The
            calibrator must be initialized when passed to this class.
        This class essentially provides the objective function for BO/RL/any other
          prompt optimizer.
        """
        super().__init__()
        if torch.cuda.is_available():
            if gpu_id:
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        self.args = args
        self.task_lm = task_lm
        if is_mask_lm is None:
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if "bert" in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm
        assert reward_type in ["gap", "cross_entropy", "entropy"]
        self.reward_type = reward_type
        print("Task LM:", self.task_lm)
        if self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
            self._generator = AutoModelForMaskedLM.from_pretrained(self.task_lm).to(
                self.device
            )
        else:
            self._generator = T5ForConditionalGeneration.from_pretrained(
                self.task_lm
            ).to(self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.task_lm, use_fast=False
            )

        self.compute_zscore = compute_zscore
        self.incorrect_coeff = incorrect_coeff
        self.correct_coeff = correct_coeff
        self.num_classes = num_classes
        print("Num classes:", self.num_classes)
        self.verbalizers = verbalizers
        print("Verbalizers:", self.verbalizers)
        self.verbalizer_ids = [
            self._tokenizer.convert_tokens_to_ids(v) for v in self.verbalizers
        ]
        print("Verbalizer ids:", self.verbalizer_ids)
        if template is None:
            self.template = self.load_default_template()  # prompt templates
        else:
            self.template = template
        self.use_bn_calibration = use_bn_calibration
        self.bn_calibrator = bn_calibrator
        self._counter = 0

    def to(self, device):
        self._generator.to(device)

    def load_default_template(self) -> List[str]:
        template_dict = {
            "xnli": [
                " {prompt} {sentence_1} {sentence_2} Entailment: ",  
                " {prompt}. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: ",  
            ],
            "mnli": [
                " {prompt} {sentence_1} {sentence_2} Entailment: ",
                " {prompt}. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: ",  
            ],
            "snli": [
                " {prompt} {sentence_1} {sentence_2} Entailment: ",
                " {prompt}. In this task, the goal is to predict textual entailment with 'yes' 'maybe' 'no'. sentence A implies sentence B entailment: yes; sentence A is neutral to sentence B entailment: maybe; sentence A contradicts sentence B entailment: no. Sentence A: {sentence_1}, Sentence B: {sentence_2}, Entailment: ",
            ],
            "rte": [
                " {prompt}. Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Textual Entailment: ",
            ],
            "sst2": [
                " {prompt}. Sentence: {sentence_1}, Sentiment: ",
            ],
            "mrpc": [
                " {prompt}. Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Semantically Equivalent: ",
            ],
            "qnli": [
                " {prompt}. Question: {sentence_1}, Sentence: {sentence_2}, Entailment: ",
            ],
            "qqp": [
                " {prompt}. Sentence 1: {sentence_1}, Sentence 2: {sentence_2}, Semantically Equivalent: ",
            ],
            "ag_news": [
                " {prompt}. Classify the news articles into the categories of World, Sports, Business, and Technology. {sentence_1}: ",
                "{prompt}\n\n{sentence_1}\n\nWhich topic is this article about?\nWorld, Sports, Business, Technology, ",
            ],
        }
        if "anli" in self.args["dataset_name"]:
            template = template_dict["anli"][self.args["template_id"]]
        elif (
            "xnli" in self.args["dataset_name"]
            or "americas_nli" in self.args["dataset_name"]
        ):
            template = template_dict["xnli"][self.args["template_id"]]
        else:
            if self.args["dataset_name"] in template_dict:
                template = template_dict[self.args["dataset_name"]][
                    self.args["template_id"]
                ]
        if self.is_mask_lm:
            mask_token = self._tokenizer.mask_token
            print(mask_token)
            simple_list = ["SetFit/sst2", "SetFit/CR", "rotten_tomatoes", "SetFit/sst5"]
            long_list = ["yelp_polarity", "yelp_review_full"]
            hard_list = ["ag_news"]
            rl_list = [
                "rl-agnews",
                "rl-cr",
                "rl-mr",
                "rl-sst-2",
                "rl-sst-5",
                "rl-yelp-2",
                "rl-yelp-5",
            ]
            if self.args["dataset_name"] in simple_list:
                template = f" {{prompt}} {{sentence_1}} It was {mask_token}."
            elif self.args["dataset_name"] in long_list:
                template = f" {{prompt}} It was {mask_token}. {{sentence_1}}"
            elif self.args["dataset_name"] in hard_list:
                template = f" {{prompt}} {mask_token} News: {{sentence_1}}"
            elif self.args["dataset_name"] in rl_list:
                template = f" {{prompt}} {{sentence_1}} It was {mask_token}."
        return template

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def forward(
        self,
        source_texts: List[str],
        source_2_texts: List[str],
        class_labels: List[int],
        output_tokens: Union[List[List[str]], List[str], str],
        # output_token: Union[List[str], str],
        to_tensor: bool,
        mode: str = "train",
        verbose: bool = True,
        accumulate_class: bool = False,
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        """
        This computes the reward of the current prompt.
        source_texts: a list of string. Usually samples from the validation set
        class_labels: a list of integers. Usually the labels of the validation set
        prompts:
          Either List[List[str]]: List of tokens. The length of the list should be the same as the number of source_texts.
          OR List[str]: List of (decoded) prompts.
          OR: str. A single prompt
        """
        assert mode in ["train", "infer"]
        if mode == "train":
            self._counter += 1

        # Process prompts and verbalizer indices
        if isinstance(output_tokens, list):
            if isinstance(output_tokens[0], list):
                prompt_tokens = output_tokens
                prompt_strings = self._convert_tokens_to_string(prompt_tokens)
            elif isinstance(output_tokens[0], str):
                prompt_strings = output_tokens
        elif isinstance(output_tokens, str):
            prompt_strings = [output_tokens]  # Single prompt string

        rewards: List[torch.Tensor] = []
        accs: List[float] = []
        confs: List[float] = []
        entropies: List[float] = []
        class_logits: List[torch.Tensor] = []

        counter_list = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log = {}
        for i, prompt in enumerate(prompt_strings):
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(
                source_texts, source_2_texts, current_prompts
            )
            all_logits = self._get_logits(formatted_templates)
            (
                reward,
                acc,
                correct_predictions,
                conf,
                entropy,
                class_logit,
            ) = _compute_reward(
                all_logits,
                target=class_labels,
                reward_type=self.reward_type,
                verbalizer_ids=self.verbalizer_ids,
                correct_coeff=self.correct_coeff,
                incorrect_coeff=self.incorrect_coeff,
                bn_calibrator=self.bn_calibrator if self.use_bn_calibration else None,
            )

            rewards.append(reward)
            accs.append(acc.item())
            confs.append(conf.item())
            entropies.append(entropy.item())
            counter_list.append(correct_predictions)
            class_logits.append(class_logit)

            # keep track of rewards for z-score normalization
            input_rewards["z"] += [reward.item()]

            # Print examples
            if verbose:
                print_strs = [
                    "Accuracy:",
                    acc.item(),
                    "|",
                    "Reward:",
                    round(reward.item(), 2),
                ]
                print(*print_strs)
        rewards_tensor = torch.stack(rewards)
        accs_tensor = torch.tensor(accs)
        confs_tensor = torch.tensor(confs)
        entropies_tensor = torch.tensor(entropies)
        # compute the expected calibration error (ECE) by accs_tensor and confs_tensor
        ece = torch.abs(accs_tensor - confs_tensor).mean()

        # z-score normalization (2nd stage)
        if mode == "train" and self.compute_zscore:
            input_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
            input_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
            # not source strings
            idx_means = torch.tensor(input_reward_means["z"]).float()
            idx_stds = torch.tensor(input_reward_stds["z"]).float()
            rewards_tensor = (rewards_tensor - idx_means) / (idx_stds + 1e-4)
            quantities_to_log[prompt_strings[i]]["resized_reward"] = []
            for i in range(rewards_tensor.size(0)):
                quantities_to_log[prompt_strings[i]]["resized_reward"].append(
                    rewards_tensor[i].item()
                )
        elif mode == "infer":  # Optional: Predict Val Prompts
            score = rewards_tensor.mean().item()
            if verbose:
                print(f"Our prompt: {prompt_strings}. Score={score}. Acc={acc}")
                for pt in prompt_strings:
                    print(self._tokenizer.tokenize(pt))
                print(accumulate_class)
                print("counter_list", counter_list)
                print("ece", ece)
                if accumulate_class:
                    return (
                        prompt_strings,
                        rewards_tensor,
                        accs_tensor,
                        counter_list,
                        ece,
                        entropies_tensor,
                        class_logits,  # <- list of tensors. n elements = n prompts
                    )
                else:
                    return prompt_strings, rewards_tensor, accs_tensor

        if to_tensor is True:
            return rewards_tensor, accs_tensor, quantities_to_log
        else:
            return rewards_tensor.tolist(), accs, quantities_to_log

    def kl_divergence_row_by_row(self, p, q):
        kl_div = torch.sum(p * torch.log(p / q), dim=1)
        return kl_div

    def compute_default_kl(
        self,
        source_texts: List[str],
        source_2_texts: List[str],
        class_labels: List[int],
        output_tokens: Union[List[List[str]], List[str], str],
        to_tensor: bool,
    ) -> torch.Tensor:
        """
        This computes the probs of the naive prompt (instruction).
        source_texts: a list of string. Usually samples from the validation set
        class_labels: a list of integers. Usually the labels of the validation set
        prompts:
          Either List[List[str]]: List of tokens. The length of the list should be the same as the number of source_texts.
          OR List[str]: List of (decoded) prompts.
          OR: str. A single prompt
        """
        default_templates = self._format_prompts(
            source_texts, source_2_texts, ["" for _ in source_texts]
        )
        default_logits = self._get_logits(default_templates)
        default_probs = _compute_probs(
            default_logits,
            target=class_labels,
            reward_type=self.reward_type,
            verbalizer_ids=self.verbalizer_ids,
            correct_coeff=self.correct_coeff,
            incorrect_coeff=self.incorrect_coeff,
        )
        return default_probs

    def compute_default_reward(
        self,
        source_texts: List[str],
        source_2_texts: List[str],
        class_labels: List[int],
        output_tokens: Union[List[List[str]], List[str], str],
        to_tensor: bool,
    ) -> torch.Tensor:
        """
        This computes the rewards of the naive prompt (instruction).
        source_texts: a list of string. Usually samples from the validation set
        class_labels: a list of integers. Usually the labels of the validation set
        prompts:
          Either List[List[str]]: List of tokens. The length of the list should be the same as the number of source_texts.
          OR List[str]: List of (decoded) prompts.
          OR: str. A single prompt
        """
        default_templates = self._format_prompts(
            source_texts, source_2_texts, ["" for _ in source_texts]
        )
        default_logits = self._get_logits(default_templates)
        default_reward, _, _, _, _, _ = _compute_reward(
            default_logits,
            target=class_labels,
            reward_type=self.reward_type,
            verbalizer_ids=self.verbalizer_ids,
            correct_coeff=self.correct_coeff,
            incorrect_coeff=self.incorrect_coeff,
        )
        return default_reward

    def compute_kl(
        self,
        source_texts: List[str],
        source_2_texts: List[str],
        class_labels: List[int],
        output_tokens: Union[List[List[str]], List[str], str],
        to_tensor: bool,
        default_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        This computes the kl-divergence of the current prompt to the naive prompt (instruction).
        source_texts: a list of string. Usually samples from the validation set
        class_labels: a list of integers. Usually the labels of the validation set
        prompts:
          Either List[List[str]]: List of tokens. The length of the list should be the same as the number of source_texts.
          OR List[str]: List of (decoded) prompts.
          OR: str. A single prompt
        """
        # Process prompts and verbalizer indices
        if isinstance(output_tokens, list):
            if isinstance(output_tokens[0], list):
                prompt_tokens = output_tokens
                prompt_strings = self._convert_tokens_to_string(prompt_tokens)
            elif isinstance(output_tokens[0], str):
                prompt_strings = output_tokens
        elif isinstance(output_tokens, str):
            prompt_strings = [output_tokens]  # Single prompt string

        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        for i, prompt in enumerate(prompt_strings):
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(
                source_texts, source_2_texts, current_prompts
            )
            all_logits = self._get_logits(formatted_templates)
            prompt_probs = _compute_probs(
                all_logits,
                target=class_labels,
                reward_type=self.reward_type,
                verbalizer_ids=self.verbalizer_ids,
                correct_coeff=self.correct_coeff,
                incorrect_coeff=self.incorrect_coeff,
            )
            kl = self.kl_divergence_row_by_row(prompt_probs, default_probs)
            kl = torch.sum(kl)
            rewards.append(kl)
        kl_tensor = torch.stack(rewards)
        return kl_tensor

    def compute_reward_diff(
        self,
        source_texts: List[str],
        source_2_texts: List[str],
        class_labels: List[int],
        output_tokens: Union[List[List[str]], List[str], str],
        to_tensor: bool,
        default_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        This computes the kl-divergence of the current prompt to the naive prompt (instruction).
        source_texts: a list of string. Usually samples from the validation set
        class_labels: a list of integers. Usually the labels of the validation set
        prompts:
          Either List[List[str]]: List of tokens. The length of the list should be the same as the number of source_texts.
          OR List[str]: List of (decoded) prompts.
          OR: str. A single prompt
        """
        # Process prompts and verbalizer indices
        if isinstance(output_tokens, list):
            if isinstance(output_tokens[0], list):
                prompt_tokens = output_tokens
                prompt_strings = self._convert_tokens_to_string(prompt_tokens)
            elif isinstance(output_tokens[0], str):
                prompt_strings = output_tokens
        elif isinstance(output_tokens, str):
            prompt_strings = [output_tokens]  # Single prompt string

        rewards: List[torch.Tensor] = []
        for i, prompt in enumerate(prompt_strings):
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(
                source_texts, source_2_texts, current_prompts
            )
            all_logits = self._get_logits(formatted_templates)
            prompt_rewards, _, _, _, _, _ = _compute_reward(
                all_logits,
                target=class_labels,
                reward_type=self.reward_type,
                verbalizer_ids=self.verbalizer_ids,
                correct_coeff=self.correct_coeff,
                incorrect_coeff=self.incorrect_coeff,
            )
            reward_diff = prompt_rewards - default_rewards
            reward_diff = torch.sum(reward_diff)
            rewards.append(reward_diff)
        reward_diff_tensor = torch.stack(rewards)
        return reward_diff_tensor

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def ensure_exactly_one_mask_token(
        self, model_inputs: Dict[str, torch.Tensor]
    ) -> None:
        for input_ids in model_inputs["input_ids"]:
            masked_index = self._get_mask_token_index(input_ids)
            numel = np.prod(masked_index.shape)
            assert numel == 1

    @torch.no_grad()
    def _get_logits(self, texts: List[str]) -> torch.Tensor:
        # for MLM, add mask token
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        decoder_input_ids = (
            torch.ones((batch_size, 1)) * torch.tensor(self._tokenizer.pad_token_id)
        ).int()
        if self.is_mask_lm:
            # self.ensure_exactly_one_mask_token(encoded_inputs) TODO
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            mask_token_indices = self._get_mask_token_index(encoded_inputs["input_ids"])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
            return out_logits
        else:
            token_logits = self._generator(
                input_ids=encoded_inputs["input_ids"].to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
            ).logits
            token_logits = token_logits[:, 0, :]
        return token_logits

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._tokenizer.convert_tokens_to_string(s) for s in tokens]

    def _format_prompts(
        self,
        source_strs: List[str],
        source_2_strs: List[str],
        prompt_strs: List[str],
    ) -> List[str]:
        return [
            self.template.format(sentence_1=s_1, sentence_2=s_2, prompt=p)
            for s_1, s_2, p in zip(source_strs, source_2_strs, prompt_strs)
        ]
