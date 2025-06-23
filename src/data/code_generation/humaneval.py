from datasets import load_dataset

from .abst_code_gen import CodeGenDataset



class HumanEvalZero_Data(CodeGenDataset):
    def __init__(self, ds, data_name):
        self.data_name = data_name
        self._ds = ds
        new_data = [transformed for d in self._ds if (transformed := self.init_transform(d)) is not None]

        super(HumanEvalZero_Data, self).__init__(self.data_name, new_data)


class HumanEvalPlusZero_Data(CodeGenDataset):
    def __init__(self):
        self.data_name = 'HumanEvalPlusZero'
        data = load_dataset("HeyixInn0/Reorganized-humanevalplus", split="train")
        new_data = []
        for item in data:
            item['demos'] = []
            new_data.append(item)
        new_data = [transformed for d in new_data if (transformed := self.init_transform(d)) is not None]

        super(HumanEvalPlusZero_Data, self).__init__(self.data_name, new_data)



# class HumanEval_Zero_Data(GenGenDataset):
#     def __init__(self, data_dir):
#         self.data_name = "HumanEvalZero"
#
#         dataset = load_dataset("openai_humaneval", split='test')
#         dataset = dataset.to_list()
#         for d in dataset:
#             data_id = d['task_id'].split('/')[1]
#             with open(f"{data_dir}/{data_id}.txt", 'r') as file:
#                 filtered_prompt = '\n'.join(file.readlines())
#             d['prompt'] = filtered_prompt
#             d['task_id'] = data_id
#
#         new_data = []
#         for d in dataset:
#             new_data.append(self.init_transform(d))
#
#         super(HumanEval_Zero_Data, self).__init__(self.data_name, new_data)
#
#     def init_transform(self, item):
#         import_str = ""
#         test_cases = item["test"] + f'check({item["entry_point"]})'
#         return CodeTask(
#             dataset_name=self.data_name,
#             data_id=item["task_id"],
#             src_lang="nl",
#             tgt_lang="python",
#             task_name="code_generation",
#             prefix=move_imports_to_top(item["prompt"]),
#             suffix="",
#             solution=move_imports_to_top(item["prompt"] + item["canonical_solution"]),
#             demos=[],
#             test_cases=[],
#             import_str=import_str,
#             entry_func=item['entry_point'],
#             entry_code="",
#             task_file_contents={},
#         )


