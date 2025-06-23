import ast
import json
from datasets import load_dataset

from .abst_code_gen import CodeGenDataset
from ..utils import split_function_signature


TAB = "    "
class MBPP_Data(CodeGenDataset):
    def __init__(self, ds, data_name):
        self.data_name = data_name

        # new_data = load_dataset("json", data_files="new_mbpp.json", split="train")
        self._ds = ds
        # new_data = []
        # for item in data:
        #     signature, body = split_function_signature(item["solution"], item['entry_func'])
        #     prompt = signature + '\n' + f'{TAB}"""\n{TAB}{item["prefix"]}\n{TAB}"""\n'
        #     item['prefix'] = prompt
        #     solution = prompt + body
        #     item['solution'] = solution
        #     new_data.append(item)
        # with open("new_mbpp.json", "w", encoding="utf-8") as f:
        #     for item in new_data:
        #         json.dump(item, f, ensure_ascii=False)
        #         f.write('\n')
        new_data = [transformed for d in  self._ds  if (transformed := self.init_transform(d)) is not None]
        super(MBPP_Data, self).__init__(self.data_name, new_data)

class MBPPPlus_Data(CodeGenDataset):
    def __init__(self):
        self.data_name = 'MBPPPlus'
        data = load_dataset("HeyixInn0/Reorganized-mbpp_plus", split="train")
        new_data = [transformed for d in data if (transformed := self.init_transform(d)) is not None]

        super(MBPPPlus_Data, self).__init__(self.data_name, new_data)