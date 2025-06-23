import ast
import re
from typing import List


class TestCase:
    def __init__(
            self,
            input_str,
            output_str,
            test_type,
            cost_time=0
    ):

        self.input_str = input_str
        self.output_str = output_str
        self.test_type = test_type

        self.cost_time = cost_time


import ast


def extract_docstring(source_code: str, func_name: str):
    """
    Extracts the docstring from a function with the given name in the provided source code.

    Args:
        source_code (str): The Python source code containing the function.
        func_name (str): The name of the function whose docstring to extract.

    Returns:
        str: The extracted docstring, or None if not found.
    """
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return ast.get_docstring(node)
        return None
    except SyntaxError as e:
        print(f"SyntaxError while parsing: {e}")
        return None


TAB = "    "
def format_and_extract_examples(docstring, entry_point):
    # Remove examples from docstring before parsing
    docstring_cleaned = re.sub(
        rf'>>>\s*{entry_point}\(.*?\)\n\s*[^>\n]+',
        '', docstring, flags=re.DOTALL)
    docstring_cleaned = docstring_cleaned.replace("\n", '')

    # Extract examples specific to the given function name
    example_pattern = re.findall(
        rf'>>>\s*{entry_point}\(([^\n]+)\)\n(\s*[^>\n]+)', docstring, flags=re.DOTALL)
    examples = [(f"{entry_point}({inp.strip()})", out.strip()) for inp, out in example_pattern]
    examples_str = '\n'.join([f"{TAB}>>>{inp}\n{TAB}{out}" for inp, out in examples])
    return docstring_cleaned.strip(), examples_str


class CodeTask:
    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            src_lang: str,
            tgt_lang: str,
            prefix: str,
            suffix: str,
            solution: str,
            demos: List[TestCase],
            test_cases: List[TestCase],
            import_str: str,
            entry_func: str,
            entry_code: str,
            task_file_contents: dict,
            **kwargs
    ):
        self.dataset_name = dataset_name
        self.data_id = data_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.test_cases = test_cases
        self.demos = demos

        self.prefix = prefix
        self.suffix = suffix
        self.solution = solution

        self.import_str = import_str
        self.entry_func = entry_func
        self.entry_code = entry_code
        self.task_file_contents = task_file_contents

        self.docstring = extract_docstring(self.solution, self.entry_func)

        self.instruction, self.demo_str = format_and_extract_examples(self.docstring, self.entry_func)

        # self.demo_single_test_str_list = ['']
        # self.demo_test_str = ""
        #
        # self.eval_single_test_str_list = [""]
        # self.eval_test_str = ""

    def __str__(self):
        return self.dataset_name + self.SPLIT + self.src_lang + self.SPLIT + self.tgt_lang + self.SPLIT +self.data_id

    def __eq__(self, other):
        if not isinstance(other, CodeTask):
            return NotImplemented
        return str(self) == str(other)

    def __getitem__(self, key):
        return getattr(self, key)


