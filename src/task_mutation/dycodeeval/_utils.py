import astor
import ast

from collections import Counter
import re


def remove_non_english_characters(input_str: str):
    """
    Removes all non-English characters from the beginning and end of the string.

    Args:
        input_str (str): The input string.

    Returns:
        str: The cleaned string with only English characters and spaces at the beginning and end.
    """
    # Use regex to remove all non-English characters from the start and end of the string
    cleaned_str = re.sub(r'^[^a-zA-Z\s]+|[^a-zA-Z\s]+$', '', input_str)
    return cleaned_str


def abstract_input_types(inferred_types_list):
    """
    Abstracts and generalizes function input types from multiple examples.

    Args:
        inferred_types_list (list): A list of tuples with inferred types for each function call.

    Returns:
        tuple: A tuple representing the generalized function input types.
    """
    num_args = max([len(d) for d in inferred_types_list])
    inferred_types_list = [d for d in inferred_types_list if len(d) == num_args]
    num_args = len(inferred_types_list[0])  # Number of function arguments
    abstracted_types = []

    for i in range(num_args):
        # Collect all types observed in the same argument position
        types_at_position = [types[i] for types in inferred_types_list]
        type_counts = Counter(types_at_position)

        # If all occurrences are the same, take that type; otherwise, generalize
        if len(type_counts) == 1:
            abstracted_types.append(types_at_position[0])
        else:
            abstracted_types.append(f"mixed({', '.join(sorted(set(types_at_position)))})")

    return tuple(abstracted_types)


def extract_input_variables(code, function_name):
    """
    Extract input variables passed to a specific function from the code.

    Args:
        code (str): The code containing the function calls.
        function_name (str): The name of the function to extract variables for.

    Returns:
        list: A list of tuples representing the arguments used in the specified function calls.
    """
    class InputExtractor(ast.NodeVisitor):
        def __init__(self):
            self.inputs = []

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                # Extract arguments from the specified function call

                self.inputs.append(tuple([eval(ast.unparse(arg)) for arg in node.args]))

            self.generic_visit(node)

    # Parse the code into an AST
    tree = ast.parse(code)

    # Use the visitor to extract inputs
    extractor = InputExtractor()
    extractor.visit(tree)

    return extractor.inputs


def extract_function_variables(code_str: str, function_name: str):
    """
    Extracts the variable names (parameters) of a specified function from a code snippet.

    Args:
        code_str (str): The source code as a string.
        function_name (str): The name of the function whose variable names are to be extracted.

    Returns:
        list: A list of variable names (parameters) of the specified function.
    """
    tree = ast.parse(code_str)
    variables = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Extract the argument names from the function definition
            variables = [arg.arg for arg in node.args.args]
            break

    return variables


def infer_detailed_types(inputs):
    """
    Infers the detailed types of function input values, including element types
    for collections like lists, sets, and dictionaries.

    Args:
        inputs (list): A list of function input tuples.

    Returns:
        list: A list of inferred detailed types for each input tuple.
    """
    def get_type(value):
        if isinstance(value, list):
            element_types = {get_type(item) for item in value}
            return f"list of {', '.join(sorted(element_types))}"
        elif isinstance(value, set):
            element_types = {get_type(item) for item in value}
            return f"set of {', '.join(sorted(element_types))}"
        elif isinstance(value, dict):
            key_types = {get_type(k) for k in value.keys()}
            value_types = {get_type(v) for v in value.values()}
            return f"dict of {', '.join(sorted(key_types))} to {', '.join(sorted(value_types))}"
        return type(value).__name__

    inferred_types = []

    for input_tuple in inputs:
        if not isinstance(input_tuple, tuple):
            input_tuple = (input_tuple,)
        types = tuple(get_type(value) for value in input_tuple)
        inferred_types.append(types)

    func_input_types = abstract_input_types(inferred_types)
    return func_input_types


# def update_docstring_with_instruction(code: str, new_instruction: str) -> str:
#     """
#     Replace the docstring of the function in the provided code snippet with the new natural language instruction.
#
#     Args:
#         code (str): Original code snippet.
#         new_instruction (str): New natural language instruction to replace the docstring.
#
#     Returns:
#         str: Modified code snippet with the updated docstring.
#     """
#
#     class DocstringUpdater(ast.NodeTransformer):
#         def visit_FunctionDef(self, node):
#             if ast.get_docstring(node):
#                 docstring = ast.get_docstring(node)
#                 lines = docstring.split('\n')
#
#                 # Extract test cases if present
#                 test_cases = [line for line in lines if line.strip().startswith('>>>')]
#
#                 # Combine new instructions with test cases
#                 updated_docstring = new_instruction.strip()
#                 if test_cases:
#
#                     updated_docstring += '\n' + '\n'.join(f'    {line}' for line in test_cases) + '\n    '
#
#                 node.body[0] = ast.Expr(value=ast.Constant(value=updated_docstring))
#             return node
#
#     # Parse the original code into an AST
#     tree = ast.parse(code)
#
#     # Update the docstring using the custom NodeTransformer
#     updated_tree = DocstringUpdater().visit(tree)
#
#     # Convert the updated AST back to source code
#     return astor.to_source(updated_tree)
#

def replace_function_docstring(source_code: str, func_name: str, new_docstring: str) -> str:
    class DocstringReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == func_name:
                node.body[0] = ast.Expr(value=ast.Constant(value=new_docstring))
            return node

    tree = ast.parse(source_code)
    tree = DocstringReplacer().visit(tree)
    return ast.unparse(tree)
def extract_data(text, tag_name):

    # Regular expression to extract content within <scenario> tags
    pattern = fr"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches