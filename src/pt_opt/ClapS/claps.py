
import random
import numpy as np
import json
import argparse
import os
import torch
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, set_seed
from typing import List
import logging
from transformers import AutoTokenizer




from .algs.genetics import GeneticAlgorithmTrainer, Genetics
from .algs.particle_swarm import ParticleSwarmOptimizer
from .algs.greedy import GreedyTrainer
from .rewards.text_classification_reward import PromptedClassificationReward
from .utils.fsc_datasets import PromptedClassificationDataset


from ..abst_pt import AbstPt
from ...data.io import CodeTask


def remove_special_token(text: str, special_token: str) -> str:
    return text.replace(special_token, "")


def find_kl_dict(args, data, vocab, obj_func, prompted_dataset):
    premise_texts, hypothesis_texts, class_labels = prompted_dataset.get_data(data)
    if args["prune_type"] == "kl":
        default_probs = obj_func.compute_default_kl(
            premise_texts, hypothesis_texts, class_labels, "", True
        )
    else:
        default_probs = obj_func.compute_default_reward(
            premise_texts, hypothesis_texts, class_labels, "", True
        )
    collect_kl = []
    kl_dict = {}
    for v, k in tqdm(vocab.items()):
        if args["prune_type"] == "kl":
            kl = obj_func.compute_kl(
                premise_texts, hypothesis_texts, class_labels, v, True, default_probs
            )
        else:
            kl = obj_func.compute_reward_diff(
                premise_texts, hypothesis_texts, class_labels, v, True, default_probs
            )
        collect_kl.append(kl)
        kl_dict[v] = kl
    for k, v in kl_dict.items():
        kl_dict[k] = float(v)
    with open(args["dict_path"], "w") as fp:
        json.dump(kl_dict, fp, indent=4, ensure_ascii=False)
    collect_kl_np = []
    for tensor in collect_kl:
        collect_kl_np.append(tensor.cpu().numpy())
    return kl_dict, collect_kl_np


def load_kl_dict(args):
    # load the KL dict from json file
    with open(args["dict_path"], "r") as fp:
        kl_dict = json.load(fp)
        collect_kl_np = []
        for k, v in kl_dict.items():
            collect_kl_np.append(v)
    return kl_dict, collect_kl_np


def load_vocab(args):
    with open(args["vocab_path"], "r") as fp:
        vocab = json.load(fp)
    vocab_key = []
    vocab_id = []
    for k, v in vocab.items():
        vocab_key.append(k)
        vocab_id.append(v)
    return vocab, vocab_key, vocab_id


def action_set_pruning(args, kl_dict, collect_kl_np, vocab):
    if not args["random_prune"]:
        collect_kl_np = np.array(collect_kl_np)
        top_10_percent = np.percentile(collect_kl_np, args["percentile"])
        # filter the vocab based on the top_10_percent_idx
        new_vocab = {
            word: vocab[word]
            for word, value in kl_dict.items()
            if value > top_10_percent
        }
        vocab = new_vocab
        vocab_key = []
        vocab_id = []
        for k, v in vocab.items():
            vocab_key.append(k)
            vocab_id.append(v)
        logger.info(len(vocab_key))
    else:
        # random select 10% of the vocab
        vocab, vocab_key, vocab_id = random_pruning(args, vocab, args["percentile"])
        logger.info(len(vocab_key))
    return vocab, vocab_key, vocab_id


def random_pruning(args, vocab: dict, percent: int = 99):
    vocab_key = []
    vocab_id = []
    for k, v in vocab.items():
        vocab_key.append(k)
        vocab_id.append(v)
    length = int(len(vocab_key) * (100 - percent) / 100)
    pruned_index = random.sample(list(np.arange(len(vocab_key))), length)
    vocab_key = [vocab_key[i] for i in pruned_index]
    vocab_id = [vocab_id[i] for i in pruned_index]
    vocab = {vocab_key[i]: vocab_id[i] for i in range(len(vocab_key))}
    logger.info(len(vocab_key))
    return vocab, vocab_key, vocab_id



class ClaPsPT(AbstPt):
    def __init__(self, seed_prompts, work_dir, name, train_data, test_data):
        super().__init__(seed_prompts, work_dir, name, train_data, test_data)
        self.args = {
            "seed": 42,
            "reprune_vocab": False,
            "num_shots": 5,
            "train_batch_size": 32,
            "model_name": "bert-base-uncased",
            "is_mask_lm": "bert" in "bert-base-uncased",
            "reward_type": "some_reward",
            "bn_calibrate": False,
            "eval_batch_size": 32,
            "method": "genetic",
            "vocab_path": "none",
            "run_manual": False,
            "num_labels": 0,
            "verbalizers": [],
        }
        self.special_space = "Ġ" if self.args["is_mask_lm"] else "▁"

    def optimize(self):

        set_seed(self.args["seed"])

        logging.info("......Truncating vocab......")
        tokenizer = AutoTokenizer.from_pretrained(self.args["model_name"])
        vocab = tokenizer.get_vocab()
        special_tokens = [tokenizer.unk_token, tokenizer.pad_token, tokenizer.sep_token, tokenizer.cls_token]
        vocab = {word: index for word, index in vocab.items() if
                 word not in special_tokens and self.special_space in word}

        logging.info("Vocab length before pruning: %s", len(vocab))
        batch_size = min(self.args["train_batch_size"], len(self.train_data))
        idx = np.random.choice(len(self.train_data), batch_size, replace=False)
        data = [self.train_data[i] for i in idx]
        logging.info(f"Length of dataset = {len(data)}")

        # obj_func = PromptedClassificationReward(
        #     args=self.args,
        #     reward_type=self.args["reward_type"],
        #     task_lm=self.args["model_name"],
        #     is_mask_lm=self.args["is_mask_lm"],
        #     num_classes=self.args["num_labels"],
        #     verbalizers=self.args["verbalizers"],
        #     use_bn_calibration=self.args["bn_calibrate"],
        # )

        obj_func

        if self.args["reprune_vocab"]:
            if self.args["vocab_path"] != "none":
                vocab, _, vocab_id = load_vocab(self.args)
            kl_dict, collect_kl_np = find_kl_dict(self.args, data, vocab, obj_func, prompt_dataset)
        else:
            kl_dict, collect_kl_np = (load_kl_dict(self.args) if not self.args["run_manual"] else ({}, []))

        if not self.args["run_manual"]:
            vocab, _, vocab_id = action_set_pruning(self.args, kl_dict, collect_kl_np, vocab)
        else:
            vocab_id = list(vocab.values())

        trainer = self._get_trainer(vocab_id, tokenizer, obj_func, prompt_dataset)

        logging.info("......Training......")
        best_str_list = trainer.train(train_dataset)

    def _get_trainer(self, vocab_id, tokenizer, obj_func, prompt_dataset):
        if self.args["method"] == "genetic":
            genetics = Genetics(tokenizer, vocab_id)
            return GeneticAlgorithmTrainer(
                pop_size=128, mutate_size=64, crossover_size=64, mutate_frac=0.1, str_len=5, epochs=30, stages=1,
                n_classes=self.args["num_labels"], genetics=genetics, eval_batch_size=self.args["eval_batch_size"],
                obj_func=obj_func, prompt_dataset=prompt_dataset, use_bn_calibrator=self.args["bn_calibrate"],
                logger=logger
            )
        elif self.args["method"] == "particle_swarm":
            return ParticleSwarmOptimizer(
                pop_size=128, epochs=30, mutate_frac=0.1, str_len=5, n_classes=self.args["num_labels"],
                eval_batch_size=self.args["eval_batch_size"], obj_func=obj_func, prompt_dataset=prompt_dataset,
                use_bn_calibrator=self.args["bn_calibrate"], logger=logger, vocab_id=vocab_id,
                crossover_tokenizer=tokenizer
            )
        elif self.args["method"] == "greedy":
            return GreedyTrainer(
                crossover_tokenizer=tokenizer, obj_func=obj_func, prompt_dataset=prompt_dataset, logger=logger,
                vocab_id=vocab_id, str_len=5, n_classes=self.args["num_labels"],
                eval_batch_size=self.args["eval_batch_size"]
            )
        else:
            raise NotImplementedError(f"Unknown method = {self.args['method']}!")


# def main(args):
#     print(args)
#     set_seed(args["seed"])
#     revocab_flag = args["reprune_vocab"]
#     shots = args["num_shots"]
#     batch_size = args["train_batch_size"]
#     args["is_mask_lm"] = False
#     special_space = "▁"
#     if "bert" in args["model_name"]:
#         args["is_mask_lm"] = True
#         special_space = "Ġ"
#     logging.info("......Loading dataset......")
#     prompt_dataset = PromptedClassificationDataset(args)
#     verbalizer_predefined = prompt_dataset.get_verbalizer()
#     args["verbalizers"] = verbalizer_predefined
#     logging.info("verbalizers: %s", verbalizer_predefined)
#     args["num_labels"] = len(verbalizer_predefined)
#     train_dataset, val_dataset, test_dataset = prompt_dataset.get_few_shot_dataset(
#         shots
#     )
#
#     logging.info("......truncating vocab......")
#     crossover_tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
#     vocab = crossover_tokenizer.get_vocab()
#     # preprocess the vocab
#     special_tokens = [
#         crossover_tokenizer.unk_token,
#         crossover_tokenizer.pad_token,
#         crossover_tokenizer.sep_token,
#         crossover_tokenizer.cls_token,
#     ]
#     vocab = {
#         word: index
#         for word, index in vocab.items()
#         if word not in special_tokens and special_space in word
#     }
#     for v in verbalizer_predefined:
#         if v not in vocab:
#             print("verbalizer not in vocab: ", v)
#         assert v in vocab
#     logging.info("the vocab length before action set pruning: %s", len(vocab))
#     dataset = train_dataset
#     print(dataset)
#     batch_size = min(batch_size, len(dataset))
#     idx = np.random.choice(len(dataset), batch_size, replace=False)
#     data = [dataset[i] for i in idx]
#     logging.info(f"Length of dataset = {len(data)}")
#     obj_func = PromptedClassificationReward(
#         args=args,
#         reward_type=args["reward_type"],
#         task_lm=args["model_name"],
#         is_mask_lm=args["is_mask_lm"],
#         num_classes=args["num_labels"],
#         verbalizers=args["verbalizers"],
#         use_bn_calibration=args["bn_calibrate"],
#     )
#
#     if revocab_flag:
#         # pruning efficiency section
#         # random select 10% of the vocab
#         if args["vocab_path"] != "none":
#             # this is to do kmeans clustering and pruning
#             vocab, _, vocab_id = load_vocab(args)
#         kl_dict, collect_kl_np = find_kl_dict(
#             args, data, vocab, obj_func, prompt_dataset
#         )
#     else:
#         if not args["run_manual"]:
#             kl_dict, collect_kl_np = load_kl_dict(args)
#         else:
#             kl_dict = {}
#             collect_kl_np = []
#     if not args["run_manual"]:
#         vocab, _, vocab_id = action_set_pruning(args, kl_dict, collect_kl_np, vocab)
#     else:
#         vocab_id = [v for k, v in vocab.items()]
#
#     if args["method"] == "genetic":
#         genetics = Genetics(crossover_tokenizer, vocab_id)
#         trainer = GeneticAlgorithmTrainer(
#             pop_size=128,
#             mutate_size=64,
#             crossover_size=64,
#             mutate_frac=0.1,
#             str_len=5,
#             epochs=30,
#             stages=1,
#             n_classes=args["num_labels"],
#             genetics=genetics,
#             eval_batch_size=args["eval_batch_size"],
#             obj_func=obj_func,
#             prompt_dataset=prompt_dataset,
#             use_bn_calibrator=args["bn_calibrate"],
#             logger=logger,
#         )
#     elif args["method"] == "particle_swarm":
#         trainer = ParticleSwarmOptimizer(
#             pop_size=128,
#             epochs=30,
#             mutate_frac=0.1,
#             str_len=5,
#             n_classes=args["num_labels"],
#             eval_batch_size=args["eval_batch_size"],
#             obj_func=obj_func,
#             prompt_dataset=prompt_dataset,
#             use_bn_calibrator=args["bn_calibrate"],
#             logger=logger,
#             vocab_id=vocab_id,
#             crossover_tokenizer=crossover_tokenizer,
#         )
#     elif args["method"] == "greedy":
#         trainer = GreedyTrainer(
#             crossover_tokenizer=crossover_tokenizer,
#             obj_func=obj_func,
#             prompt_dataset=prompt_dataset,
#             logger=logger,
#             vocab_id=vocab_id,
#             str_len=5,
#             n_classes=args["num_labels"],
#             eval_batch_size=args["eval_batch_size"],
#         )
#     else:
#         raise NotImplementedError(f"Unknown method = {args['method']}!")
#     if not args["run_manual"]:
#         logging.info("......training......")
#         best_str_list = trainer.train(train_dataset)
#


#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--method",
#         type=str,
#         default="genetic",
#         choices=["genetic", "particle_swarm", "greedy"],
#     )
#     parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
#     parser.add_argument("--dataset_name", type=str, default="xnli")
#     parser.add_argument(
#         "--reward_type",
#         type=str,
#         default="cross_entropy",
#         help="cross_entropy or entropy",
#     )
#     parser.add_argument("--prune_type", type=str, default="reward", help="reward or kl")
#     parser.add_argument("--num_shots", type=int, default=16)
#     parser.add_argument("--train_batch_size", type=int, default=2000)
#     parser.add_argument("--eval_batch_size", type=int, default=8)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument(
#         "--n_epochs", type=int, default=30, help="Number of training epochs"
#     )
#     parser.add_argument(
#         "--percentile", type=float, default=99, help="top x% of the tokens to prune"
#     )
#     parser.add_argument(
#         "--reprune_vocab",
#         type=bool,
#         default=False,
#         help="whether to prune again for the complete vocab",
#     )
#     parser.add_argument(
#         "--random_prune", type=bool, default=False, help="whether to prune randomly"
#     )
#     parser.add_argument(
#         "--ngram_prune", type=bool, default=False, help="whether to prune ngrams"
#     )
#     parser.add_argument(
#         "--run_manual",
#         action="store_true",
#         help="whether to evaluate the manual template",
#     )
#     parser.add_argument("--save_path", type=str, default="output")
#     parser.add_argument("--dict_path", type=str, default="./kl_dict.json")
#     parser.add_argument("--vocab_path", type=str, default="none")
#     parser.add_argument(
#         "--template_id",
#         type=int,
#         default=0,
#         help="the index for the prompt template to be evaluated",
#     )
#     parser.add_argument("--bn_calibrate", action="store_true")
#     parser.add_argument("--save_logits", action="store_true")
#
#     args = parser.parse_args()
#     args = vars(args)
#     if not os.path.exists(args["save_path"]):
#         os.makedirs(args["save_path"])
#     logger.addHandler(
#         logging.FileHandler(os.path.join(args["save_path"], "output.log"))
#     )
#     main(args)
