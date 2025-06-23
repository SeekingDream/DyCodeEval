from typing import List
import ast
import os
import torch
from ..utils import MyDataset
from ..utils import move_imports_to_top
from ..io import CodeTask, TestCase


class CodeGenDataset(MyDataset):
    def __init__(self, data_name, dataset: List[CodeTask]):
        super().__init__(data_name, dataset)
        self.task_name = "code_gen"

    def init_transform(self, item):

        # try:
        input_list, output_list = [], []
        for d in item['test_cases']:
            input_list.append(d[0])
            output_list.append(d[1])

        input_tuple_list = [eval("(" + str(i) + ",)") for i in input_list]
        output_list = [eval(str(i)) for i in output_list]
        entry_code = item['compare_func'].format(
            input_list=input_tuple_list,
            output_list=output_list,
            func_entry=item['entry_func'],
        )
        return CodeTask(
            dataset_name=self.data_name,
            data_id=str(item["data_id"]).replace('/', "_"),
            src_lang=str(item["src_lang"]),
            tgt_lang=item["tgt_lang"],
            task_name=item["task_name"],
            prefix=move_imports_to_top(item["prefix"]),
            suffix=item["suffix"],
            solution=move_imports_to_top(item["solution"]),
            demos=[TestCase(d[0], d[1], None) for d in item['demos']],
            test_cases=[TestCase(d[0], d[1], None) for d in item['test_cases']],
            import_str="\n".join(item['import_str']),
            entry_func=item['entry_func'],
            entry_code=entry_code,
            task_file_contents={},
        )
        # except Exception as e:
        #     print(self.data_name,  item["data_id"], e, "transfer error")
        #     return None



class SynCodeGen_Data(CodeGenDataset):
    def __init__(self, save_dir, llm_name):
        self.SPLIT_SYM = "____SPLIT____"
        new_data_list = []
        import warnings
        for file_name in os.listdir(save_dir):
            save_path = os.path.join(save_dir, file_name)

            warnings.filterwarnings("ignore")
            tmp = torch.load(save_path, weights_only=False)
            tmp = tmp[:1]
            for i, d in enumerate(tmp):
                data_point = d.to_dict()
                data_point['data_id'] = str(data_point['data_id']) + f"{self.SPLIT_SYM}{i}"
                data_point['dataset_name'] = f"syn{self.SPLIT_SYM}{llm_name}{self.SPLIT_SYM}" + str(
                    data_point['dataset_name'])
                new_data_point = CodeTask(
                    dataset_name= data_point['dataset_name'],
                    data_id=data_point['data_id'],
                    src_lang="nl",
                    tgt_lang=data_point['lang'],
                    prefix=data_point['prefix'],
                    suffix=data_point['suffix'],
                    solution=data_point['solution'],
                    demos=[],
                    test_cases=[],
                    import_str=data_point['import_str'],
                    entry_func=data_point['entry_point'],
                    entry_code=data_point['test_case_str'],
                    task_file_contents= {}
                )

                new_data_list.append(new_data_point)

        self.data_name = new_data_list[0]['dataset_name']
        super(SynCodeGen_Data, self).__init__(self.data_name, new_data_list)