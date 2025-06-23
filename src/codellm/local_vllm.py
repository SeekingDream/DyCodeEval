from typing import List
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import time
from copy import deepcopy
from vllm.lora.request import LoRARequest
from vllm.distributed import destroy_distributed_environment
from vllm.distributed import destroy_model_parallel
import gc
import contextlib
import torch

from .llm_abst import AbstLLM


class LocalVLLM(AbstLLM):
    def __init__(self, provider, model, llm_config: dict):
        super().__init__(provider, model, llm_config)

        self.logprobs = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.stop_token_ids = [self.tokenizer.eos_token_id]

        tp_size = llm_config.get("tp_size", 1)
        max_model_len = llm_config.get("max_model_len", 2048)
        is_lora = llm_config.get("is_lora", False)
        dtype = llm_config.get("dtype", "auto")
        self.llm = LLM(
            model=model,
            enforce_eager=True,
            trust_remote_code=True,
            dtype = dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=tp_size,
            enable_lora=is_lora,
            disable_custom_all_reduce=True
        )

        self.sampling_params = None
        self.dtype = None
        self.lora_request = None


    def init_ai_kwargs(self, config):
        super().init_ai_kwargs(config)
        lora_path = config.get('lora_path', None)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.logprobs,
            stop=self.stop
        )

        if lora_path is not None:
            self.lora_request = LoRARequest("tmp", 1, lora_path)
        self.is_init = True

    @staticmethod
    def extract_text_logprobs(model_pred):
        def extract_token_prob(d_list):
            res = []
            for d in d_list:
                tmp = list(d.values())[0]
                res.append([tmp.decoded_token, tmp.logprob])
            return res

        text = model_pred.outputs[0].text
        logprobs = extract_token_prob(model_pred.outputs[0].logprobs)
        return text, logprobs

    def cleanup(self,):
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()

    def chat_llm(self, messages):
        # messages = [[{"content": pt, "role": "user"}] for pt in prompts]
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request
        )
        return outputs

    def competition_llm(self, prompts: List[str]):
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request
        )
        return outputs

# class LocalCompletionVLLM(LocalVLLM):
#     def __init__(self, provider, model):
#         super().__init__(provider, model)
#


#
# class LocalChatVLLM(LocalVLLM):
#     def __init__(self, provider, model):
#         super().__init__(provider, model)



