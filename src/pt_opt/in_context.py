import numpy as np
from typing import List
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from supar import Parser
import copy

from .abst_pt import AbstPtOptimizer, AbstPt
from ..data.io import CodeTask
from ..evaluator import AbstEvaluator


class InContextLearningOptimizer(AbstPtOptimizer):
    def __init__(
            self,
            seed_pts: List[AbstPt],
            code_llm,
            agent,
            evaluator: AbstEvaluator,
            train_data: List[CodeTask],
            valid_data: List[CodeTask],
            test_data: List[CodeTask],
            config: dict
    ):
        self.num_demos = config['num_demos']
        self.demos = train_data[:self.num_demos]
        super().__init__(
            seed_pts, code_llm, agent, evaluator, train_data, valid_data, test_data, config
        )

    def optimize_pt(self):
        new_pt = copy.deepcopy(self.seed_pts[0])
        new_pt.demos = self.demos
        return {"best_pt": new_pt}
