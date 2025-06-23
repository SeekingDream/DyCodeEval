import numpy as np
from .base_trainer import BaseTrainer
from utils.fsc_datasets import PromptedClassificationDataset
from rewards.text_classification_reward import PromptedClassificationReward
from typing import Any
import copy


class ParticleSwarmOptimizer(BaseTrainer):
    def __init__(
        self,
        pop_size: int,
        epochs: int,
        mutate_frac: float,
        str_len: int,
        n_classes: int,
        eval_batch_size: int,
        obj_func: PromptedClassificationReward,
        prompt_dataset: PromptedClassificationDataset,
        logger: Any,
        use_bn_calibrator: bool,
        vocab_id,
        crossover_tokenizer,
    ):
        super().__init__(
            obj_func=obj_func,
            prompt_dataset=prompt_dataset,
            logger=logger,
            use_bn_calibrator=use_bn_calibrator,
        )
        self.crossover_tokenizer = crossover_tokenizer
        self.vocab_id = vocab_id
        self.pop_size = pop_size
        self.epochs = epochs
        self.mutate_frac = mutate_frac
        self.str_len = str_len
        self.n_classes = n_classes
        self.eval_batch_size = eval_batch_size

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def predict_batch(
        self,
        sentences,
    ):
        return np.array(
            [
                self.predict(
                    s,
                )
                for s in sentences
            ]
        )

    def predict(
        self,
        sentence,
    ):
        # Alia for reward computation -- note that we expect
        # a list of int in terms of vocab_id for sentence argument here.
        sentence_str = self.crossover_tokenizer.decode(
            sentence, skip_special_tokens=True
        )
        tem = (
            self.obj_func.forward(
                self.premise_texts,
                self.hypothesis_texts,
                self.class_labels,
                [sentence_str],
                True,
                "infer",
                verbose=False,
            )[0]
            .detach()
            .cpu()
            .item()
        )

        return tem

    def select_best_replacement(self, pos, x_cur, replace_list):
        """Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list"""
        new_x_list = [
            self.do_replace(x_cur, pos, w) if w != 0 else x_cur for w in replace_list
        ]
        # Randomly select some rather than enumerate, which is very slow
        new_x_list_str = [
            self.crossover_tokenizer.decode(s, skip_special_tokens=True)
            for s in new_x_list
        ]
        x_scores = (
            self.obj_func.forward(
                self.premise_texts,
                self.hypothesis_texts,
                self.class_labels,
                new_x_list_str,
                True,
                "infer",
                verbose=False,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        # new_x_preds = self.predict_batch(new_x_list)
        # x_scores = new_x_preds  # [:, target]
        orig_score = self.predict(x_cur)  # [target]

        new_x_scores = x_scores - orig_score
        # Eliminate not that clsoe words

        if np.max(new_x_scores) > 0:
            best_id = np.argsort(new_x_scores)[-1]
            return [x_scores[best_id], new_x_list[best_id]]
        return [orig_score, x_cur]

    def perturb(self, x_cur, neigbhours, w_select_probs):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        # while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(
        #     np.sign(w_select_probs)
        # ):
        #     rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        replace_list = neigbhours[rand_idx]
        x_cur[rand_idx] = np.random.choice(replace_list)
        score = self.predict(x_cur)
        return [score, x_cur]
        # return self.select_best_replacement(rand_idx, x_cur, replace_list)

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def equal(self, a, b):
        return -3 if a == b else 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def train(self, train_data):
        (
            self.premise_texts,
            self.hypothesis_texts,
            self.class_labels,
        ) = self.prompt_dataset.get_data(train_data)

        neigbhours_list = [self.vocab_id for _ in range(self.str_len)]
        neighbours_len = [len(x) for x in neigbhours_list]
        x_len = self.str_len
        #
        w_select_probs = []
        for pos in range(x_len):
            if neighbours_len[pos] == 0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos], 10))
        w_select_probs = w_select_probs / np.sum(w_select_probs)

        if np.sum(neighbours_len) == 0:
            return None

        # Generate random population
        pop = [
            np.random.choice(self.vocab_id, self.str_len) for _ in range(self.pop_size)
        ]
        pop_scores = self.predict_batch(
            pop,
        )

        part_elites = copy.deepcopy(pop)
        part_elites_scores = pop_scores
        all_elite_score = np.max(pop_scores)
        pop_ranks = np.argsort(pop_scores)
        top_attack = pop_ranks[-1]
        all_elite = pop[top_attack]

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for rrr in range(self.pop_size)]
        V_P = [[V[t] for rrr in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.epochs):
            Omega = (Omega_1 - Omega_2) * (self.epochs - i) / self.epochs + Omega_2
            C1 = C1_origin - i / self.epochs * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.epochs * (C1_origin - C2_origin)

            for id in range(self.pop_size):
                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                        self.equal(pop[id][dim], part_elites[id][dim])
                        + self.equal(pop[id][dim], all_elite[dim])
                    )
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)

            pop_scores = []
            pop_scores_all = []
            for a in pop:
                pt = self.predict(a)

                pop_scores.append(pt)
                pop_scores_all.append(pt)
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]

            if self.logger is not None:
                self.logger.info(
                    f"{i}  --  {pop_scores[top_attack]}。 Best = {self.crossover_tokenizer.decode(all_elite, add_special_tokens=False)}"
                )

            new_pop = []
            new_pop_scores = []
            for id in range(len(pop)):
                x = pop[id]
                if np.random.uniform() < self.mutate_frac:
                    tem = self.perturb(x, neigbhours_list, w_select_probs)
                    # if tem is None:
                    #     return None
                    # # if tem[0] == 1:
                    # # return tem[1]
                    # else:
                    new_pop_scores.append(tem[0])
                    new_pop.append(tem[1])
                else:
                    new_pop_scores.append(pop_scores[id])
                    new_pop.append(x)
            pop = new_pop

            pop_scores = new_pop_scores
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]
            for k in range(self.pop_size):
                if pop_scores[k] > part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            elite = pop[top_attack]
            if np.max(pop_scores) > all_elite_score:
                all_elite = elite
                all_elite_score = np.max(pop_scores)

        all_elite_str = self.crossover_tokenizer.decode(
            all_elite, add_special_tokens=False
        )

        return [all_elite_str]
