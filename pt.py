
from src.pt_opt.code_pt import ChatCodeGenPT, CompetitionCodeGenPT
from src.pt_opt.code_pt import INST_POSTION

chat_code_gen_pt = ChatCodeGenPT(
    instruction="You are a helpful coding assistant producing high-quality code. "
                "Strictly follow the given docstring and function signature {position_word} "
                "to complete the function. Your code should always gracefully return. "
                "Your response should include all dependencies, headers and function declaration to be directly usable "
                "(even for the ones seen in the given part). "
                "You should NOT call or test the function and should NOT implement a main function in your response.\n\n"
                "You should implement the function in Python. "
                "You should output your complete code implementation in a single code block wrapped by triple backticks.",
    data_template="Code:\n```\n{prefix}\n```",
    name="chat_cg",
    position=INST_POSTION.PREVIOUS
)

complete_code_gen_pt = CompetitionCodeGenPT(
    instruction="",
    data_template="{prefix}",
    name="complete_cg",
    position=INST_POSTION.PREVIOUS
)

CODE_GEN_PT_DICT = {
    "chat": chat_code_gen_pt,
    "competition": complete_code_gen_pt
}

