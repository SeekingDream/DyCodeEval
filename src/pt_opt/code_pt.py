from typing import List
import re
from .abst_pt import AbstPt
from ..data.io import CodeTask
from .abst_pt import INST_POSTION


class ChatCodeGenPT(AbstPt):
    def __init__(
        self, instruction, data_template, name, position,
    ):
        super().__init__(instruction, data_template, name, position)

    def task2pt(self, task: CodeTask) -> str:
        if self.position == 'PREVIOUS':
            position_word = "below"
        else:
            position_word = "above"
        keys = re.findall(r'\{([^}]+)\}', self.instruction)
        map_dict = {
            "src_lang": str(task.src_lang),
            "tgt_lang": str(task.tgt_lang),
            "position_word": str(position_word)
        }
        for k in keys:
            if k not in map_dict.keys():
                map_dict[k] = k
        try:
            instruction_str = self.instruction.format(**map_dict)
        except:
            instruction_str = self.instruction
            # print(map_dict)
            # print(self.instruction)
            # print('--------------------------ERROR-------------------')

        data_str = self.data_template.format(
            prefix=str(task.prefix),
            src_lang=str(task.src_lang),
            tgt_lang=str(task.tgt_lang),
            suffix=str(task.suffix)
        )
        if self.position == INST_POSTION.PREVIOUS:
            pt = instruction_str + '\n\n' + data_str
        else:
            pt = data_str + '\n\n' + instruction_str
        return pt

    def extract_ans(self, prompt_str, llm_output_str):
        return self.extract_code_block(llm_output_str)


class CompetitionCodeGenPT(AbstPt):
    def __init__(
        self, instruction, data_template, name, position,
    ):
        super().__init__(instruction, data_template, name, position)

    def task2pt(self, task: CodeTask) -> str:
        return task.prefix

    def extract_ans(self, prompt_str, llm_output_str):
        return prompt_str + llm_output_str
