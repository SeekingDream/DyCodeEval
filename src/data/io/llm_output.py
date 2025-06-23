import ast

from .utils import LLMReqRec, SPLIT_STRING
from .code_gen_task import CodeTask


class CodeLLMOutput:
    def __init__(
            self,
            llm_req_rec: LLMReqRec,
            final_ans,
            cost_time,
    ):
        self.__dict__.update(vars(llm_req_rec))

        self.llm_req_rec = llm_req_rec
        self.final_ans = final_ans
        self.cost_time = cost_time

        self.data_id = llm_req_rec.ori_task.data_id
    @property
    def ori_task(self):
        return self.llm_req_rec.ori_task

    @property
    def prompt_input(self):
        return self.llm_req_rec.prompt_input

    @property
    def llm_output(self):
        return self.llm_req_rec.llm_output

    @property
    def sampling_params(self):
        return self.llm_req_rec.sampling_params

    @property
    def text(self):
        return self.llm_req_rec.text

    @property
    def logits(self):
        return self.llm_req_rec.logits

        # self.code_with_test = self.make_test_code()
        # self.code_with_single_test_list = self.make_single_test_code()
        # self.is_correct = None

    # def make_test_code(self):
    #     import_str = self.original_task.import_str
    #     test_case_str = self.original_task.eval_test_str
    #     code_with_test = (import_str + f'\n\n{SPLIT_STRING}\n\n'
    #                       + self.final_code + f'\n\n{SPLIT_STRING}\n\n'
    #                       + test_case_str)
    #     return code_with_test
    #
    # def make_single_test_code(self):
    #     all_test_codes = []
    #     import_str = self.original_task.import_str
    #     for test_case_str in self.original_task.eval_single_test_str_list:
    #         code_with_test = (import_str + f'\n\n{SPLIT_STRING}\n\n'
    #                           + self.final_code + f'\n\n{SPLIT_STRING}\n\n'
    #                           + test_case_str)
    #         all_test_codes.append(code_with_test)
    #     return all_test_codes
    #
    # @property
    # def is_parseable(self):
    #     if self.original_task.lang.lower() == "python":
    #         try:
    #             ast.parse(self.final_code)
    #             return True
    #         except Exception as e:
    #             return False
    #     else:
    #         raise NotImplemented
    #
    # def check_is_correct(self, code_str, test_cases_str):
    #
    #     raise NotImplementedError