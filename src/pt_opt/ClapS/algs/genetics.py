import random
import numpy as np
from typing import Any
from .base_trainer import BaseTrainer
from utils.fsc_datasets import PromptedClassificationDataset
from rewards.text_classification_reward import PromptedClassificationReward


class Genetics:
    def __init__(self, crossover_tokenizer, vocab_id):
        self.crossover_tokenizer = crossover_tokenizer
        self.vocab_id = vocab_id

    def mutate(self, x, prob=0.1):
        """
        Mutates the input string by replacing tokens with a certain probability.

        Args:
            x (str): The input string.
            prob (float, optional): The probability of replacing each token. Defaults to 0.1.

        Returns:
            str: The mutated string.
        """
        x_list = self.crossover_tokenizer.encode(x)

        def pick_another(x_, candidates):
            return (
                x_
                if len(candidates) == 1
                else random.choice([v for v in candidates if v != x_])
            )

        for i, element in enumerate(x_list):
            if i == 0 or i == len(x_list) - 1:
                continue
            if random.random() < prob:
                x_list[i] = pick_another(element, self.vocab_id)

        out = self.crossover_tokenizer.decode(x_list, skip_special_tokens=True)
        return out

    def crossover(self, x1, x2):
        """
        Performs crossover between two input strings.

        Args:
            x1 (str): The first input string.
            x2 (str): The second input string.

        Returns:
            str: The crossover result.
        """

        def _crossover_helper(v1, v2):
            return v1 if random.random() < 0.5 else v2

        def _inbalance_helper(v1, v2):
            n_tokens = min(len(v1), len(v2))
            max_n = max(len(v1), len(v2))
            out_token = []
            for i in range(n_tokens):
                out_token.append(v1[i] if random.random() < 0.5 else v2[i])
            for i in range(n_tokens, max_n):
                out_token.append(v1[i] if len(v1) > n_tokens else v2[i])
            return out_token

        x1_tokens = self.crossover_tokenizer.encode(x1)
        x2_tokens = self.crossover_tokenizer.encode(x2)
        x = _crossover_helper(x1_tokens, x2_tokens)
        ret = self.crossover_tokenizer.decode(x, skip_special_tokens=True)
        return ret

    def random_string(self, length=5):
        """
        Generates a random string of a specified length.

        Args:
            length (int, optional): The length of the random string. Defaults to 5.

        Returns:
            str: The random string.
        """
        choices = self.vocab_id
        out = random.choices(choices, k=length)
        out = self.crossover_tokenizer.decode(out, skip_special_tokens=True)
        return out

    def random_extend_pop(self, pop: list, n: int) -> list:
        """
        Extends the population with random strings.

        Args:
            pop (list): The population.
            n (int): The number of random strings to generate.

        Returns:
            list: The extended population.
        """
        pop = [p + self.random_string(n) for p in pop]
        return pop


class GeneticAlgorithmTrainer(BaseTrainer):
    def __init__(
        self,
        pop_size: int,
        mutate_size: int,
        crossover_size: int,
        epochs: int,
        mutate_frac: float,
        str_len: int,
        stages: int,
        n_classes: int,
        eval_batch_size: int,
        genetics: Genetics,
        obj_func: PromptedClassificationReward,
        prompt_dataset: PromptedClassificationDataset,
        logger: Any,
        use_bn_calibrator: bool,
    ):
        super().__init__(
            obj_func=obj_func,
            prompt_dataset=prompt_dataset,
            logger=logger,
            use_bn_calibrator=use_bn_calibrator,
        )
        self.pop_size = pop_size
        self.mutate_size = mutate_size
        self.crossover_size = crossover_size
        self.epochs = epochs
        self.mutate_frac = mutate_frac
        self.str_len = str_len
        self.stages = stages
        self.n_classes = n_classes
        self.genetics = genetics
        self.epoch_per_extend = 3
        self.extend_size = 128
        self.eval_batch_size = eval_batch_size

    def train(self, train_data):
        premise_texts, hypothesis_texts, class_labels = self.prompt_dataset.get_data(
            train_data
        )
        epoch_per_stage = self.epochs // self.stages
        start_str = ""
        best_str_list = []

        for _ in range(self.stages):
            pop = [
                self.genetics.random_string(self.str_len) for _ in range(self.pop_size)
            ]
            if self.logger is not None:
                self.logger.info(pop)
            old_reward = 0
            epoch_counter = 0
            for evo_epoch in range(epoch_per_stage):
                if self.str_len == 1:
                    pop_ = [start_str + " " + p for p in pop]
                else:
                    pop_ = [start_str + p for p in pop]
                reward = self.obj_func.forward(
                    premise_texts,
                    hypothesis_texts,
                    class_labels,
                    pop_,
                    True,
                    "infer",
                    verbose=False,
                )[0]
                if self.logger is not None:
                    self.logger.info(
                        f"Epoch = {evo_epoch}. Max reward = {reward.max()}. Best prompt = {pop_[reward.argmax()]}"
                    )
                max_reward = reward.max()
                if max_reward > old_reward:
                    old_reward = max_reward
                    epoch_counter = 0
                else:
                    epoch_counter += 1

                sorted_idx = reward.argsort(descending=True)[
                    : max(1, int(reward.shape[0] * self.mutate_frac))
                ]
                pop = [pop[i] for i in sorted_idx]
                mutate_cfgs, crossover_cfgs = [], []
                extend_cfgs = []
                for _ in range(self.mutate_size):
                    old_cfg = np.random.choice(pop)
                    cfg = self.genetics.mutate(old_cfg)
                    mutate_cfgs.append(cfg)

                for _ in range(self.crossover_size):
                    cfg1 = np.random.choice(pop)
                    cfg2 = np.random.choice(pop)
                    cfg = self.genetics.crossover(cfg1, cfg2)
                    crossover_cfgs.append(cfg)

                pop += mutate_cfgs
                pop += crossover_cfgs

                if self.logger is not None:
                    self.logger.info(
                        f"Epoch = {evo_epoch}. Population length = {len(pop)}"
                    )

                if self.str_len > 1:
                    if pop[reward.argmax()] not in best_str_list:
                        best_str_list.append(pop[reward.argmax()])
                else:
                    if pop_[reward.argmax()] not in best_str_list:
                        best_str_list.append(pop_[reward.argmax()])
                    # if we do step by steo do the pop_
            if self.str_len == 1:
                pop_ = [start_str + " " + p for p in pop]
            else:
                pop_ = [start_str + p for p in pop]
            start_str = pop_[reward.argmax()]

        return best_str_list

    def random_train(self, train_data):
        premise_texts, hypothesis_texts, class_labels = self.prompt_dataset.get_data(
            train_data
        )
        start_str = ""
        best_str_list = []
        pop = [
            self.genetics.random_string(self.str_len)
            for _ in range(self.pop_size * self.epochs)
        ]
        # logger.info(pop)
        pop_ = [start_str + p for p in pop]
        reward = self.obj_func.forward(
            premise_texts,
            hypothesis_texts,
            class_labels,
            pop_,
            True,
            "infer",
            verbose=False,
        )[0]

        if self.logger is not None:
            self.logger.info(
                f"Max reward = {reward.max()}. Best prompt = {pop_[reward.argmax()]}"
            )
        if pop[reward.argmax()] not in best_str_list:
            best_str_list.append(pop[reward.argmax()])
        return best_str_list
