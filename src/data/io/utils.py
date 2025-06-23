from .code_gen_task import CodeTask

SPLIT_STRING = "###### SPLIT STRING #####"


class LLMReqRec:
    def __init__(
            self,
            ori_task: CodeTask,
            prompt_input,
            llm_output,
            sampling_params,
            text,
            logits,
    ):
        self.ori_task = ori_task
        self.prompt_input = prompt_input
        self.llm_output = llm_output
        self.sampling_params = sampling_params
        self.text = text
        self.logits = logits




