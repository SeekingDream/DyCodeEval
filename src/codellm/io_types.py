import ast
import re
import docstring_parser
from typing import List, Tuple, Union, Dict

from .prompt_utils import extract_docstrings_and_clean_code
from .prompt_utils import replace_function_docstring

TAB = "    "
class CodeTask:
    dataset_name: str
    data_id: str
    lang: str
    task_name: str
    prefix: str
    suffix: str
    solution: str
    test_case_str: str
    import_str: str
    entry_point: str
    # config: Union[Dict, None]
    def init(self):
        # TODO
        self.old_docstring, _ = self.split_docstring(self.old_prefix)
        _, self.clean_code = self.split_docstring(self.solution)
        self.instruction, self.demo_str = self.format_and_extract_examples(self.old_docstring)
        multiple_inst = self.instruction.split('. ')
        multiple_inst = [d.strip() for d in multiple_inst]
        multiple_inst = f'.\n{TAB}'.join(multiple_inst)
        self.demo_str = self.demo_str.strip()
        self.docstring = f'\n{TAB}{multiple_inst}\n{TAB}{self.demo_str}\n{TAB}'
        self.docstring = self.docstring.rstrip() + f"\n{TAB}"
        self.prefix = replace_function_docstring(
            self.old_prefix, self.entry_point, self.docstring) + '\n'
        self.solution = replace_function_docstring(
            self.solution, self.entry_point, self.docstring)
        return self

    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            lang: str,
            task_name: str,
            prefix: str,
            suffix: str,
            solution: str,
            test_case_str: str,
            import_str: str,
            entry_point: str
            # config: Union[Dict, None] = None,
    ):
        self.dataset_name = dataset_name
        self.data_id = data_id
        self.lang = lang
        assert lang.lower() in ['python', 'c']
        self.task_name = task_name

        assert task_name in ['CC', 'CG']

        self.old_prefix = prefix
        self.suffix = suffix
        self.solution = solution
        self.test_case_str = test_case_str
        self.import_str = import_str
        self.entry_point = entry_point
        self.init()


    def split_docstring(self, code_str) -> Tuple[str, str]:
        docstring_dict, cleaned_code = extract_docstrings_and_clean_code(code_str)
        # assert len(docstrings) == 1
        return docstring_dict[self.entry_point], cleaned_code

    
    def __str__(self):
        return self.dataset_name + '::::' + self.lang + "::::" + self.data_id + "::::" + self.cwe

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "data_id": self.data_id,
            "lang": self.lang,
            "task_name": self.task_name,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "solution": self.solution,
            "test_case_str": self.test_case_str,
            "import_str": self.import_str,
            "entry_point":self.entry_point,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create an instance of the class from a dictionary.

        Parameters:
            data (dict): A dictionary with keys corresponding to the class attributes.

        Returns:
            MyClass: An instance of the class with attributes populated from the dictionary.
        """
        # Use dictionary unpacking to initialize attributes
        data = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        data.init()
        return data

    def __eq__(self, other):
        if not isinstance(other, CodeTask):
            return NotImplemented
        return str(self) == str(other)

    def __getitem__(self, key):
        return getattr(self, key)

    def format_and_extract_examples(self, docstring):
        # Remove examples from docstring before parsing
        docstring_cleaned = re.sub(
            rf'>>>\s*{self.entry_point}\(.*?\)\n\s*[^>\n]+',
            '', docstring, flags=re.DOTALL)
        docstring_cleaned = docstring_cleaned.replace("\n", '')
        
        # Extract examples specific to the given function name
        example_pattern = re.findall(
            rf'>>>\s*{self.entry_point}\(([^\n]+)\)\n(\s*[^>\n]+)', docstring, flags=re.DOTALL)
        examples = [(f"{self.entry_point}({inp.strip()})", out.strip()) for inp, out in example_pattern]
        examples_str = '\n'.join([f"{TAB}>>>{inp}\n{TAB}{out}" for inp, out in examples ])
        return docstring_cleaned.strip(), examples_str


class CodeLLMOutput:
    def __init__(
            self,
            prompt_input,
            original_task: CodeTask,
            original_output,
            text,
            logits,
            final_code,
            cost_time
    ):
        self.prompt_input = prompt_input
        self.original_task = original_task
        self.text = text
        self.logits = logits
        self.original_output = original_output
        self.final_code = final_code
        self.cost_time = cost_time

    def is_parseable(self):
        if self.original_task.lang.lower() == "python":
            try:
                ast.parse(self.final_code)
                return True
            except Exception as e:
                return False
        else:
            raise NotImplemented

