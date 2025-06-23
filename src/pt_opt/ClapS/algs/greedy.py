import random
import numpy as np
from typing import Any, Optional

from rewards.text_classification_reward import (
    PromptedClassificationReward,
)
from utils.fsc_datasets import PromptedClassificationDataset
from .base_trainer import BaseTrainer
from utils.fsc_datasets import PromptedClassificationDataset
from rewards.text_classification_reward import PromptedClassificationReward


class GreedyTrainer(BaseTrainer):
    def __init__(
        self,
        obj_func: PromptedClassificationReward,
        prompt_dataset: PromptedClassificationDataset,
        vocab_id,
        crossover_tokenizer,
        str_len: int,
        n_classes: int,
        eval_batch_size: int,
        logger,
        use_bn_calibrator: bool = False,
        n_samples_bn_calibrator: int = 128,
    ):
        super().__init__(
            obj_func, prompt_dataset, logger, use_bn_calibrator, n_samples_bn_calibrator
        )
        self.vocab_id = vocab_id
        self.crossover_tokenizer = crossover_tokenizer
        self.str_len = str_len
        self.n_classes = n_classes
        self.eval_batch_size = eval_batch_size

    def train(self, train_data):
        premise_texts, hypothesis_texts, class_labels = self.prompt_dataset.get_data(
            train_data
        )
        prompt = ""
        candidate_strs = [
            self.crossover_tokenizer.decode([d], skip_special_tokens=True)
            for d in self.vocab_id
        ]
        for _ in range(self.str_len):
            pop = [prompt + candidate_str for candidate_str in candidate_strs]
            # Evaluate the reward of all pop
            reward = (
                self.obj_func.forward(
                    premise_texts,
                    hypothesis_texts,
                    class_labels,
                    pop,
                    True,
                    "infer",
                    verbose=False,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            best_reward_idx = np.argmax(reward)
            if not prompt:
                prompt = candidate_strs[best_reward_idx]
            else:
                prompt += candidate_strs[best_reward_idx]
            print(f"Current reward = {reward[best_reward_idx]}. Best prompt = {prompt}")
        return [prompt]
