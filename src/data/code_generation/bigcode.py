import json
from datasets import load_dataset

from .abst_code_gen import CodeGenDataset
from ..io import CodeTask, TestCase
from ..utils import move_imports_to_top, extract_func_tests
from ..utils import extract_docstrings_and_clean_code


class BigCode_Data(CodeGenDataset):

    def transform(self, item):
        solution_code = item['complete_prompt'] + item['canonical_solution']
        entry_code = (item['test'] + "\n\n" +
                      "if __name__ == '__main__':\n\n"
                      "    unittest.main()")
        # test_case_info = extract_func_tests(item['test'], item['entry_point'])

        inst = item['instruct_prompt']
        entry_func = item['entry_point']
        doc_struct = json.loads(item['doc_struct'])
        doc_struct_example = [d.replace('>>>', "").strip() for d in doc_struct['examples']]
        doc_struct_example = [d for d in doc_struct_example if d.find(entry_func)!=-1]

        demo_case_info = [
            res for example in doc_struct_example
            if (res := extract_func_tests(example, entry_func)) is not None
        ]
        import_str = ""
        try:
            task =  CodeTask(
                dataset_name=self.data_name,
                data_id=str(item["task_id"]).replace('/', "_"),
                src_lang="nl",
                tgt_lang="python",
                task_name="code_generation",
                prefix=inst,
                suffix="",
                solution=move_imports_to_top(solution_code),
                demos=[],
                test_cases=[TestCase(str(info[0]), "", None) for info in demo_case_info],
                import_str=import_str,
                entry_func=entry_func,
                entry_code=entry_code,
                task_file_contents={},
        )
            #  docstrings, cleaned_code = extract_docstrings_and_clean_code(solution_code)
            # inst = docstrings[item['entry_point']].split("\nArgs:")[0]
            # inst = inst.split("\nParameters:")
            # if len(inst) == 1:
            #     print(inst[0] + '\n')
            #     print('----------------------------')
            task.instruction = inst
            return task
        except Exception as e:
            print(self.data_name, item['task_id'], e, "transfoer error")
            return None


    def __init__(self):
        self.data_name = 'bigcodebench'
        orig_ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")

        new_data = [transformed for d in orig_ds if (transformed := self.transform(d)) is not None]

        super(CodeGenDataset, self).__init__(self.data_name, new_data)


