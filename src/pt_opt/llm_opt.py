import numpy as np
from typing import List
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import copy

from src.pt_opt.abst_pt import AbstPtOptimizer, AbstPt
from src.data.io import CodeTask
from src.evaluator import AbstEvaluator

class LLMOptimizer(AbstPtOptimizer):
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
        self.agent = agent
        self.num_compose = 1

        super().__init__(
            seed_pts, code_llm, agent, evaluator, train_data, valid_data, test_data, config
        )

        self.mutate_template = \
            ("I have some texts along with their corresponding scores. "
             "The texts are arranged in ascending order based on their scores, "
             "where higher scores indicate better quality.\n\n"
             "{score_str}\n\n"
             "The following exemplars show how to apply your text: "
             "you replace <INS> in each input with your text, then read the input and give an output. "
             "We say your output is wrong if your output is different from the given output, "
             "and we say your output is correct if they are the same.\n\n"
             "{demo_str}\n\n"
             "Write your new text that is different from the old ones and has a score as high as possible. "
             "Write the text in <text></text> tags.")

    def optimize_pt(self):
        score_list = [(copy.deepcopy(pt), self.score_pt(pt, self.train_data)) for pt in self.seed_pts]
        best_score_list = []
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        best_pt_list = []
        for _ in range(self.num_steps):
            selected_pt_demo = score_list[:3]

            score_str_list = []
            for pt_demo in selected_pt_demo:
                score_str_list.append(
                    f"text:\n{pt_demo[0].instruction}\nscore:\n{pt_demo[1]}\n"
                )
            score_str = "\n".join(score_str_list)
            demo_str_list = []
            for task in self.train_data[:3]:
                demo_str_list.append(
                    f"Input:\n<INS>\n\n{task.prefix}\n"
                    f"You should output your complete implementation in a single code block wrapped by triple backticks.\n\n\n"
                    f"Output:\n"
                    f"Here is the code\n"
                    f"```{task.tgt_lang}\n"
                    f"{task.solution}\n"
                    f"```"
                )
            demo_str = "\n".join(demo_str_list)
            query_str = self.mutate_template.format(
                score_str=score_str,
                demo_str=demo_str
            )
            # print(query_str)
            message = [{"content": query_str, "role": "user"}]
            responses = self.agent.chat_llm([message])
            new_inst = self.seed_pts[0].extract_html_data(
                responses[0].choices[0].message.content, "text")[0]
            new_pt = copy.deepcopy(self.seed_pts[0])
            new_pt.instruction = new_inst
            new_score = self.score_pt(new_pt, self.train_data)
            print(new_score)
            score_list.append((new_pt, new_score))

            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
            best_score_list.append(score_list[0][1])
            best_pt_list.append(score_list[0][0])

        return {
            "best_pt": score_list[0][0],
            "best_pt_list": best_pt_list,
            "train_scores_list": best_score_list
        }


