import json

from src.data.utils import MyDataset
from src.codellm import CodeGenTask


class CWEEvalDataset(MyDataset):
    def __init__(self, data_name, data_path):
        self.data_name = data_name

        with open(data_path, 'r') as f:
            data = json.load(f)

        dataset = [self.init_transform(item) for item in data]

        super().__init__(data_name, dataset)

    def init_transform(self, item):
        return CodeGenTask(
            dataset_name=self.data_name,
            data_id=(item['task_dir'] + self.SPLIT_SYM + item['file_name']).replace('/', '_'),
            lang=item['language'],
            task_prompt=item['task_desc_with_lib'],
            solution=item['file_content'],
            demos=[],
            test_cases=[],
            import_str="",
            entry_code=item['entry_point'],
            entry_func="unknow",
            task_file_contents=item['task_file_contents'],
        )

    def get_save_file_name(self, task: CodeGenTask):
        save_file_name = task.data_id.split(self.SPLIT_SYM)[-1]
        return save_file_name

if __name__ == '__main__':
    data_path = "/home/sc5687/2_Project/SecCodeGen/dataset/preprocess_data/cweeval.json"
    CWEEvalDataset('cweeval', data_path)