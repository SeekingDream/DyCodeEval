from datasets import load_dataset

from ..utils import MyDataset
from ..utils import move_imports_to_top
from ..io import CodeTask, TestCase

class TransCoder_Data(MyDataset):
    def __init__(self, data_name, data_url):
        self.data_name = data_name
        new_data = load_dataset(data_url, split="train")

        new_data = [transformed for d in new_data if (transformed := self.init_transform(d)) is not None]

        super(TransCoder_Data, self).__init__(self.data_name, new_data)

    def init_transform(self, item):
        input_list, output_list = [], []
        for d in item['test_cases']:
            input_list.append(d[0])
            output_list.append(d[1])

        try:
            input_tuple_list = [eval("(" + str(i) + ",)") for i in input_list]
            output_list = [eval(str(i)) for i in output_list]
            entry_code = self.PY_TEST_CODE.format(
                input_list=input_tuple_list,
                output_list=output_list,
                func_entry=item['entry_func'],
            )
            return CodeTask(
                dataset_name=item['dataset_name'],
                data_id=str(item["data_id"]).replace('/', "_"),
                src_lang=item["src_lang"],
                tgt_lang=item["tgt_lang"],
                task_name=item["task_name"],
                prefix=item["prefix"],
                suffix=item["suffix"],
                solution=move_imports_to_top(item["solution"]),
                demos=[TestCase(d[0], d[1], None) for d in item['demos']],
                test_cases=[TestCase(d[0], d[1], None) for d in item['test_cases']],
                import_str="\n".join(item['import_str']),
                entry_func=item['entry_func'],
                entry_code=entry_code,
                task_file_contents={},
            )
        except Exception as e:
            print(self.data_name,  item["data_id"], e)
            return None

