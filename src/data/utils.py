import ast, astor
import os
import subprocess
import random
from tqdm import tqdm
from typing import List, Any
from .io import CodeTask

PY_TAB = "    "
SPLIT = "____SPLIT____"


def split_at_last_function_signature(code: str):
    # Parse the code into an Abstract Syntax Tree (AST)
    tree = ast.parse(code)

    # Collect all function definitions
    function_defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_defs.append(node)

    if not function_defs:
        raise ValueError("No function definitions found in the input code.")

    # Get the last function definition
    last_function = function_defs[-1]

    # Extract everything up to the last function signature (imports and previous functions)
    code_up_to_last_function = ''
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node != last_function:
            code_up_to_last_function += ast.unparse(node) + '\n'

    # Extract the signature of the last function definition (i.e., the 'def ...' part)
    function_signature = ast.unparse(last_function).split('\n')[0]  # Only take the first line

    # Extract the body of the last function
    function_body = '\n'.join(ast.unparse(last_function).split('\n')[1:])

    return code_up_to_last_function + function_signature, function_body

def split_function_signature(code: str, func_name):
    # Parse the code into an Abstract Syntax Tree (AST)
    tree = ast.parse(code)

    # Collect all function definitions
    function_defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == func_name:
                function_defs.append(node)

    if not function_defs:
        raise ValueError("No function definitions found in the input code.")

    assert len(function_defs) == 1
    last_function= function_defs[0]

    # Extract everything up to the last function signature (imports and previous functions)
    code_up_to_last_function = ''
    for node in tree.body:
        if node is not last_function:
            code_up_to_last_function += ast.unparse(node) + '\n'


    function_signature = ast.unparse(last_function).split('\n')[0]  # Only take the first line

    # Extract the body of the last function
    function_body = '\n'.join(ast.unparse(last_function).split('\n')[1:])

    return code_up_to_last_function + function_signature, function_body


def extract_function_names(code):
    """
    Extract all function names defined in the code and filter out built-in functions.

    Args:
        code (str): The code to analyze.

    Returns:
        list: A list of function names defined in the code.
    """

    class FunctionNameExtractor(ast.NodeVisitor):
        def __init__(self):
            self.functions = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)

    # Parse the code into an AST
    tree = ast.parse(code)

    # Extract function names
    extractor = FunctionNameExtractor()
    extractor.visit(tree)

    # Filter out built-in function names
    built_in_functions = dir(__builtins__)

    res = [func for func in extractor.functions if func not in built_in_functions]
    if len(res) == 0:
        res = extractor.functions
    return res

def move_imports_to_top(code: str) -> str:
    class ImportMover(ast.NodeTransformer):
        def __init__(self):
            self.imports = []
            super().__init__()

        def visit_Import(self, node):
            self.imports.append(node)
            return None  # Remove from original position

        def visit_ImportFrom(self, node):
            self.imports.append(node)
            return None  # Remove from original position

    tree = ast.parse(code)
    mover = ImportMover()
    tree = mover.visit(tree)
    ast.fix_missing_locations(tree)

    # Reconstruct the new module body with imports at the top
    tree.body = mover.imports + [node for node in tree.body if node not in mover.imports]

    # Convert the modified AST back to source code
    return astor.to_source(tree)

import ast
from typing import List, Tuple, Any, Optional

class TaskFuncTestCaseExtractor(ast.NodeVisitor):
    def __init__(self, target_func_name: str):
        self.target_func_name = target_func_name
        self.current_result_var = None
        self.results: List[Tuple[Any, Optional[str]]] = []  # (input, output_type)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.current_result_var = None
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Handle task_func(...) calls
        if isinstance(node.func, ast.Name) and node.func.id == self.target_func_name:
            # Try to extract argument
            if node.args:
                try:
                    arg_val = ast.literal_eval(node.args[0])
                except Exception:
                    arg_val = ast.unparse(node.args[0])
            else:
                arg_val = None

            # Check if this call is part of an assignment
            parent = getattr(node, 'parent', None)
            var_name = None
            if isinstance(parent, ast.Assign) and isinstance(parent.targets[0], ast.Name):
                var_name = parent.targets[0].id
                self.current_result_var = var_name

            self.results.append([arg_val, None])
        self.generic_visit(node)

def extract_func_tests(code: str, target_func_name: str = "task_func") -> List[Tuple[Any, Optional[str]]]:
    try:
        tree = ast.parse(code)
    except Exception as e:
        return None
    extractor = TaskFuncTestCaseExtractor(target_func_name)
    extractor.visit(tree)
    return extractor.results




class MyDataset:
    @staticmethod
    def is_input_str_tuple(input_str):
        input_v = eval(input_str)
        if input_str.strip()[0] == '(' and input_str[-1] == ')' and isinstance(input_v, tuple):
            return True
        else:
            return False

    PY_PATH = "/local/arise/CM/miniconda3/envs/codellm/bin/python"
    # PY_TEST_CODE = (
    #     "import numpy as np\n\n"
    #     "def is_floats(x) -> bool:\n"
    #     "    # check if it is float; List[float]; Tuple[float]\n"
    #     "    if isinstance(x, float):\n"
    #     "        return True\n"
    #     "    if isinstance(x, (list, tuple)):\n"
    #     "        return all(isinstance(i, float) for i in x)\n"
    #     "    if isinstance(x, np.ndarray):\n"
    #     "        return x.dtype == np.float64 or x.dtype == np.float32\n"
    #     "    return False\n\n"
    #     "def assertion(out, exp, atol):\n"
    #     "    exact_match = out == exp\n"
    #     "    if atol == 0 and is_floats(exp):\n"
    #     "        atol = 1e-6\n"
    #     "    if not exact_match and atol != 0:\n"
    #     "        assert np.allclose(out, exp, rtol=1e-07, atol=atol)\n"
    #     "    else:\n"
    #     "        assert exact_match\n\n"
    #     "inputs = {input_list}\n\n"
    #     "results = {output_list}\n\n"
    #     "candidate = {func_entry}\n\n"
    #     "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
    #     "    assertion(candidate(*inp), exp, 0)"
    # )

    def execute_script(self, file_path):

        try:
            # Run script inside exec_dir
            result = subprocess.run(
                [self.PY_PATH, os.path.abspath(file_path)],
                capture_output=True,
                text=True,
                timeout=5,  # Timeout to prevent infinite loops

            )
            return result.stdout, result.stderr, result.returncode == 0
        except Exception as e:
            print(f"Error executing {file_path}: {e}")
            return None, None, False


    def __init__(self, data_name, dataset: List[CodeTask]):
        self.dataset = dataset
        self.data_name = data_name

        # self.validate_data()

    def validate_data(self, data_sample, tmp_dir):

        os.makedirs(tmp_dir, exist_ok=True)
        code = data_sample.solution
        test = data_sample.entry_code
        final_code = code + '\n' + test

        save_path = os.path.join(tmp_dir, f"{data_sample.data_id}.py")
        with open(save_path, "w") as f:
            f.write(final_code)
        stdout, std_error, errors = self.execute_script(save_path)

        # The original logic for whether to keep this sample
        should_keep = True
        if not errors:
            print(self.data_name, data_sample.data_id, save_path)

            if isinstance(std_error, str) :
                print(stdout, std_error)
                should_keep = False


        return should_keep, stdout, std_error, save_path

    def validate_dataset(self, tmp_dir):

        new_dataset = []
        for d in tqdm(self.dataset):
            should_keep, stdout, std_error, save_path = (
                self.validate_data(d, tmp_dir))
            if should_keep:
                new_dataset.append(d)
        self.dataset = new_dataset

    def __getitem__(self, index):
        return self.dataset[index]  # Allow indexing like a list

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for item in self.dataset:
            yield item


    def random_split(self, seed, percentage_dict):
        assert sum(percentage_dict.values()) <= 1.0
        random.seed(seed)
        data_list = self.dataset[:]  # Copy the list to avoid modifying the original
        random.shuffle(data_list)  # Shuffle the list randomly

        total = len(data_list)
        split_sizes = {k: int(total * percentage_dict[k]) for k in percentage_dict}

        return_dict = {}
        start = 0
        for k in split_sizes:
            size = split_sizes[k]

            name = self.data_name + "_" + k
            end_index = min(start + size, len(data_list))
            return_dict[k] = MyDataset(name, data_list[start:end_index])
            start += size
        return return_dict


def extract_docstrings_and_clean_code(code_str: str):
    """
    Extracts docstring comments from the given Python code string and
    returns the code without docstrings.

    Args:
        code_str (str): The source code as a string.

    Returns:
        tuple: A tuple containing a list of extracted docstrings and the code without docstrings.
    """
    tree = ast.parse(code_str)
    docstrings = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            docstring = ast.get_docstring(node)

            if docstring:
                signature = node.name
                docstrings[signature] = docstring
                # Remove the docstring from the source code by replacing it
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                    node.body.pop(0)

    cleaned_code = ast.unparse(tree)

    return docstrings, cleaned_code