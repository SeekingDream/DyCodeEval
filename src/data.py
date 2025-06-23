import os
from datasets import Dataset, load_dataset
import ast
import torch
import astor

from legacy.codellm import CodeTask


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


def move_imports_to_top(code: str) -> str:
    tree = ast.parse(code)
    mover = ImportMover()
    tree = mover.visit(tree)
    ast.fix_missing_locations(tree)

    # Reconstruct the new module body with imports at the top
    tree.body = mover.imports + [node for node in tree.body if node not in mover.imports]

    # Convert the modified AST back to source code
    return astor.to_source(tree)


class MyDataset(Dataset):
    SPLIT_SYM = "____SPLIT____"

    def __init__(self, new_data: Dataset):

        super(MyDataset, self).__init__(new_data.data)

    @staticmethod
    def process_item(item):
        pass

    def __iter__(self):
        for item in self.to_list():
            yield CodeTask.from_dict(item)

    # def to_list(self) -> list:
    #     res = []
    #     for item in self.data:
    #         res.append(CodeTask.from_dict(item))
    #     return res
    def group(self):
        results = {}
        for data in self:
            k = str(data['data_id']).split(self.SPLIT_SYM)[0]
            if k not in results:
                results[k] = []
            results[k].append(data)
        return results


class HumanEvalData(MyDataset):
    def __init__(self):
        dataset = load_dataset("openai_humaneval", split='test')
        self.data_name = "HumanEval"

        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)

        super(HumanEvalData, self).__init__(new_data)

    def init_transform(self, item):
        import_str = ""
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeTask(
            dataset_name=self.data_name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prefix=move_imports_to_top(item["prompt"]),
            suffix="",
            solution=move_imports_to_top(item["prompt"] + item["canonical_solution"]),
            test_case_str=test_cases,
            import_str=import_str,
            entry_point=item['entry_point'],
        )


TAB = '    '


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


class MBPPData(MyDataset):
    def __init__(self):
        dataset = load_dataset("mbpp", "sanitized", split="test")
        self.data_name = "MBPP"
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPData, self).__init__(new_data)

    def init_transform(self, item):
        signature, body = split_at_last_function_signature(item["code"])
        prompt = signature + '\n' + f'{TAB}"""\n{TAB}{item["prompt"]}\n{TAB}"""\n'
        solution = prompt + body
        import_st = '\n'.join(item['test_imports'])
        solution = import_st + '\n' + solution
        test_case_str = "\n".join(item['test_list'])
        all_func_names = extract_function_names(prompt)
        entry_point = all_func_names[-1]

        return CodeTask(
            dataset_name=self.data_name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prefix=move_imports_to_top(prompt),
            suffix="",
            solution=move_imports_to_top(solution),
            test_case_str=test_case_str,
            import_str=import_st,
            entry_point=entry_point,
        )


class SyntheticData(MyDataset):

    def __init__(self, save_dir, llm_name):
        new_data_list = []
        import warnings
        for file_name in os.listdir(save_dir):
            save_path = os.path.join(save_dir, file_name)

            warnings.filterwarnings("ignore")
            tmp = torch.load(save_path)
            for i, d in enumerate(tmp):
                data_point = d.to_dict()
                data_point['data_id'] = str(data_point['data_id']) + f"{self.SPLIT_SYM}{i}"
                data_point['dataset_name'] = f"syn{self.SPLIT_SYM}{llm_name}{self.SPLIT_SYM}" + str(
                    data_point['dataset_name'])
                new_data_list.append(data_point)
        new_data = Dataset.from_list(new_data_list)
        self.data_name = new_data_list[0]['dataset_name']
        super(SyntheticData, self).__init__(new_data)
