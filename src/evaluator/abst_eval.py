import os
import subprocess
from typing import List
from abc import abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ..data.io import CodeLLMOutput

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



def compute_bleu(reference, candidate):
    """Compute BLEU score between reference and candidate code snippets."""
    ref_tokens = [reference.split()]
    cand_tokens = candidate.split()
    return sentence_bleu(ref_tokens, cand_tokens)

# def compute_codebleu(reference, candidate):
#     alpha,beta,gamma,theta = 0.25, 0.25, 0.25, 0.25
#
#
#
#     references = []
#     for i in range(len(hypothesis)):
#         ref_for_instance = []
#         for j in range(len(pre_references)):
#             ref_for_instance.append(pre_references[j][i])
#         references.append(ref_for_instance)
#     assert len(references) == len(pre_references)*len(hypothesis)
#
#
#     # calculate ngram match (BLEU)
#     tokenized_hyps = [x.split() for x in hypothesis]
#     tokenized_refs = [[x.split() for x in reference] for reference in references]
#
#     ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)
#
#     # calculate weighted ngram match
#     keywords = [x.strip() for x in open('keywords/'+args.lang+'.txt', 'r', encoding='utf-8').readlines()]
#     def make_weights(reference_tokens, key_word_list):
#     return {token:1 if token in key_word_list else 0.2 \
#             for token in reference_tokens}
#     tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
#                 for reference_tokens in reference] for reference in tokenized_refs]
#
#     weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)
#
#     # calculate syntax match
#     syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, args.lang)
#
#     # calculate dataflow match
#     dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, args.lang)
#
#     print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
#                         format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
#
#     code_bleu_score = alpha*ngram_match_score\
#                     + beta*weighted_ngram_match_score\
#                     + gamma*syntax_match_score\
#                     + theta*dataflow_match_score
#
#     print('CodeBLEU score: ', code_bleu_score)






class AbstEvaluator:

    def __init__(self, work_dir, num_proc, PY_PATH):
        self.work_dir = work_dir
        self.num_proc = num_proc
        self.PY_PATH = PY_PATH

    @abstractmethod
    def evaluate(self, llm_output_list: List[CodeLLMOutput]):
        pass


class ExeEvaluator(AbstEvaluator):
    def __init__(self, work_dir,  num_proc, PY_PATH):
        super().__init__(work_dir, num_proc, PY_PATH)

    def execute_script(self, file_path):
        try:
            result = subprocess.run(
                [self.PY_PATH, os.path.abspath(file_path)],
                capture_output=True,
                text=True,
                timeout=10,  # Timeout to prevent infinite loops
            )
            return result.stdout, result.stderr, result.returncode == 0
        except Exception as e:
            print(f"Error executing {file_path}: {e}")
            return None, None, False

    def evaluate(self, llm_output_list: List[CodeLLMOutput]):
        file_path_list = []
        score_list = []

        for llm_output in llm_output_list:
            import_str = llm_output.ori_task.import_str
            pred_code = llm_output.final_ans
            test_code = llm_output.ori_task.entry_code
            test_code = import_str + '\n\n' + pred_code + "\n" + test_code + '\n\n'
            num = len(os.listdir(self.work_dir))
            file_path = os.path.join(self.work_dir, f"{(num + 1)}.py")
            file = open(file_path, 'w')
            file.write(test_code)
            file.close()
            file_path_list.append(file_path)

        if self.num_proc == 1: # Sequential execution
            for file_path in file_path_list:
                _, _, is_correct = self.execute_script(file_path)
                score_list.append(is_correct)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.num_proc) as executor:
                results = executor.map(self.execute_script, file_path_list)
                score_list = [result[2] for result in results]
        return_dict = {
            "score": np.mean(score_list)
        }
        return return_dict



class BLEUEvaluator(AbstEvaluator):

    def evaluate(self, llm_output_list: List[CodeLLMOutput]):
        score_list = []
        for llm_output in llm_output_list:
            ground_truth_code = llm_output.ori_task.solution
            pred_code = llm_output.final_ans
            score = compute_bleu(ground_truth_code, pred_code)
            score_list.append(score)
        return_dict = {
            "score": np.mean(score_list)
        }
        return return_dict


# class CodeBLEUEvaluator(AbstEvaluator):
#     def evaluate(self, llm_output_list: List[CodeLLMOutput]):
#         score_list = []
#         for llm_output in llm_output_list:
#             ground_truth_code = llm_output.ori_task.solution
#             pred_code = llm_output.final_ans
#             score = XXXX(ground_truth_code, pred_code)
#             score_list.append(score)
#         return_dict = {
#             "score": np.mean(score_list)
#         }
#         return return_dict