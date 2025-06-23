import os
import re
from typing import List
import subprocess
import numpy as np
from abc import abstractmethod
from enum import Enum

from ..data.io import CodeTask


class INST_POSTION(Enum):
    PREVIOUS = 0
    AFTER = 1


class AbstPt:

    def __init__(
        self, instruction: str, data_template: str, name:str,
        position: INST_POSTION, demos=None
    ):
        if demos is None:
            demos = []
        self.instruction = instruction
        self.data_template = data_template
        self.name = name + "____" + str(position)
        self.position = position
        self.demos = demos

    def task2msg(self, task):
        msg = []
        for demo in self.demos:
            msg.extend(self.task2demo(demo))
        pt = self.task2pt(task)
        new_msg = self._pt2msg(pt)
        msg.extend(new_msg)
        return msg

    def task2pt(self, task):
        raise NotImplementedError

    def extract_ans(self, prompt_str, llm_output_str):
        return self.extract_code_block(llm_output_str)

    def _pt2msg(self, pt):
        messages = [{"content": pt, "role": "user"}]
        return messages

    def msg2pt(self, msg):
        raise NotImplementedError

    def task2demo(self, task: CodeTask):
        pt = self.task2pt(task)
        questions_message = self._pt2msg(pt)
        answer = (f"Here is the {task.tgt_lang} code.\n"
                  f"```{task.tgt_lang}\n"
                  f"{task.solution}\n"
                  f"```")
        questions_message.append({"content": answer, "role": "assistant"})
        return questions_message

    @staticmethod
    def extract_html_data(text, tag_name):
        # Regular expression to extract content within <scenario> tags
        pattern = fr"<{tag_name}>(.*?)</{tag_name}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def extract_code_block(self, code):
        match = re.search(r"```(?:\w+)?\n(.*?)\n```", code, re.DOTALL)
        return match.group(1) if match else ""



class AbstPtOptimizer:
    def __init__(
        self,
        seed_pts: List[AbstPt],
        code_llm,
        agent,
        evaluator,
        train_data: List[CodeTask],
        valid_data: List[CodeTask],
        test_data: List[CodeTask],
        config: dict
     ):
        self.seed_pts = seed_pts
        self.code_llm = code_llm
        self.agent = agent
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.evaluator = evaluator
        self.config = config

        self.num_steps = config["num_steps"]

    def score_pt(self, pt:AbstPt, tasks:List[CodeTask]):
        llm_output_list = self.code_llm.chat_batch(pt, tasks)
        return_dict = self.evaluator.evaluate(llm_output_list)
        score = return_dict['score']
        return score


    def mutate_pt(self, pt_list: List[AbstPt]):
        raise NotImplementedError

    @abstractmethod
    def optimize_pt(self):
        pass