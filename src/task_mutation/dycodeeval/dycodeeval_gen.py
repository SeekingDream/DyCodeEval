import copy
import os
import torch
import random
from tqdm import tqdm
from typing import List, Set, Dict
from openai import OpenAI
import argparse
import json
import boto3
import time
import threading
import re
import traceback

from utils import load_my_dataset
from utils import NEW_PROMPT_DIR
from ._utils import *
from ._dycodeeval_pt import SCENARIO_PT, CONTEXT_PT
from ._dycodeeval_pt import REWRITE_PT, VERIFY_PT
from ...data import CodeTask

from ...codellm import AbstLiteLLM, LocalVLLM, AbstLLM




AMZ_MODEL = 1
OPENAI_MODEL = 2

class ProblemGen:
    # def init_client(self):
    #     if self.model_type is AMZ_MODEL:
    #
    #         # Initialize Bedrock client
    #         client = boto3.client('bedrock-runtime', region_name='us-west-2')
    #     elif self.model_type is OPENAI_MODEL:
    #         openai_api_key = "EMPTY"
    #         openai_api_base = "http://localhost:8000/v1"
    #         client = OpenAI(
    #             api_key=openai_api_key,
    #             base_url=openai_api_base,
    #         )
    #     else:
    #         raise NotImplementedError
    #     return client
    #
    # def _gpt_pt2text(self, prompts, temperature, max_len):
    #     completion = self.client.completions.create(
    #         model=self.model_name,
    #         prompt=prompts,
    #         temperature=temperature,
    #         max_tokens=max_len,
    #     )
    #     return [d.text for d in completion.choices]
    #
    # def _invoke_model(self, payload, result_list, index):
    #     try:
    #         response = self.client.invoke_model(
    #             modelId=self.model_name,
    #             accept="application/json",
    #             contentType="application/json",
    #             body=json.dumps(payload),
    #         )
    #         stream = response['body'].read().decode('utf-8')
    #         res = json.loads(stream)['content'][0]['text']
    #         result_list[index] = res
    #     except self.client.exceptions.ServiceQuotaExceededException as e:
    #         time.sleep(60)
    #         self._invoke_model(payload, result_list, index)
    #     except self.client.exceptions.ThrottlingException as e:
    #         time.sleep(60)
    #         self._invoke_model(payload, result_list, index)
    #     except Exception as e:
    #         print(e)
    #         result_list[index] = None
    #
    # def _amz_pt2text(self, prompts, temperature, max_len):
    #     payload_list = [{
    #         "messages": [
    #             {"content": f"{pt}", "role": "user"}
    #         ],
    #         "max_tokens": max_len,
    #         "temperature": temperature,
    #         "anthropic_version": "bedrock-2023-05-31"  # Required key
    #     } for pt in prompts]
    #
    #     all_res = [None] * len(payload_list)  # Initialize result list with None
    #     threads = []
    #
    #     for i, payload in enumerate(payload_list):
    #         thread = threading.Thread(target=self._invoke_model, args=(payload, all_res, i))
    #         threads.append(thread)
    #         thread.start()
    #
    #     for thread in threads:
    #         thread.join()
    #     all_res = [d for d in all_res if d is not None ]
    #     return all_res

    # def prompt2text(self, prompts: List[str], temperature: float = 0.6, max_len=2048):
    #     if self.model_type is OPENAI_MODEL:
    #         return self._gpt_pt2text(prompts, temperature, max_len)
    #     elif self.model_type is AMZ_MODEL:
    #         return self._amz_pt2text(prompts, temperature, max_len)
    #     else:
    #         raise NotImplementedError

    def prompt2text(
            self, prompt_list: List[str],
            temperature: float = 0.6,
            top_p: float = 0.95,
            max_tokens=2048
    ):
        messages = [[{"content": pt, "role": "user"}] for pt in prompt_list]
        config = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'stop': []
        }
        self.agent.init_ai_kwargs(config)
        llm_output = self.agent.chat_llm(messages)
        gen_text = [self.agent.extract_text_logprobs(d)[0] for d in llm_output]
        return gen_text


    SCENARIO_SEEDS = {
        "banking", "healthcare", "education", "transportation", "social networking"
    }
    SCENARIO_EXAMPLE_LIST = ["Banking: Fraud Detection."]
    SCENARIO_TAG = "scenario"

    def __init__(
            self, agent: AbstLLM,
            work_dir: str,
            scenario_num, context_num,
    ):
        self.agent = agent
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.scenario_path = str(os.path.join(self.work_dir, f"scenario.pt"))

        self.scenario_num = scenario_num
        self.context_num = context_num


    def scenario_proposer(self):
        """
        Dynamically generates new scenarios by iteratively prompting and extracting new examples.
        The final scenarios are saved to `self.scenario_path`.
        """
        if os.path.isfile(self.scenario_path):
            with open(self.scenario_path, "r") as f:
                scenario_dict = json.load(f)
        else:
            scenario_dict = {}
        if len(scenario_dict) > self.scenario_num:
            print("Scenario num {} already exists!".format(len(scenario_dict)))
            return

        seed_prompt = copy.deepcopy(SCENARIO_PT)
        scenario_seeds: Set[str] = set(self.SCENARIO_SEEDS)
        example_list: List[str] = list(self.SCENARIO_EXAMPLE_LIST)
        scenario_dict: Dict[str, str] = {}

        def extract_and_add_scenarios(scenario_texts: List[str]):
            """
            Extract scenario strings from LLM outputs, then process and add them to the collections.
            """
            for text in scenario_texts:
                data_items = extract_data(text, tag_name=self.SCENARIO_TAG)
                if not data_items:
                    continue
                for scenario_text_ in (data_items if isinstance(data_items, list) else [data_items]):
                    try:
                        data = scenario_text_.split(':')
                        if len(data) < 2:
                            continue
                        scenario_key = remove_non_english_characters(f"{data[0].strip()}:{data[1].strip()}")
                        new_scenario = remove_non_english_characters(scenario_text_)
                        scenario_seeds.add(remove_non_english_characters(data[0].strip()))
                        example_list.append(new_scenario)
                        scenario_dict.setdefault(scenario_key, new_scenario)
                    except Exception as e:
                        print(f"[WARN] Error processing scenario '{scenario_text_}': {e}")

        def prepare_prompt_list() -> List[str]:
            """
            Constructs a list of prompts using current seeds and examples.
            """
            new_prompts = []
            for _ in range(10):
                selected_seeds = random.sample(list(scenario_seeds), 5)
                selected_example = random.choice(example_list)
                new_prompts.append(
                    seed_prompt.format(
                        s1=selected_seeds[0], s2=selected_seeds[1],
                        s3=selected_seeds[2], s4=selected_seeds[3],
                        s5=selected_seeds[4], exp=selected_example
                    )
                )
            return new_prompts

        for _ in tqdm(range(self.scenario_num)):
            prompts = prepare_prompt_list()
            raw_scenarios = self.prompt2text(prompts, temperature=0.8, max_tokens=100)
            extract_and_add_scenarios(raw_scenarios)
            if len(scenario_dict) > self.scenario_num:
                break

        with open(self.scenario_path, 'w') as f:
            json.dump(scenario_dict, f, indent=4)

    def context_proposer(self, prob_pt, input_names, func_input_types, scenario_dict):
        prompt_template = copy.deepcopy(CONTEXT_PT)

        CONTEXT_TAG = "context"
        var_str = '\n'.join([f"{n}: {t}" for n, t in zip(input_names, func_input_types)])
        assert var_str != ""
        sample_num = min(self.context_num, len(list(scenario_dict.keys())))
        scenario_list = random.sample(list(scenario_dict.keys()), k=sample_num)
        prompt_list = []
        for scenario in scenario_list:
            prompt = prompt_template.format(
                pb=prob_pt, var=var_str, scenario=scenario)
            prompt_list.append(prompt)
        llm_output = self.prompt2text(prompt_list, max_tokens=200)
        context = [extract_data(d, CONTEXT_TAG) for d in llm_output]
        context = [d[0] for d in context if len(d) > 0]
        return scenario_list, context

    def prompt_rewrite(self, ori_prompt, scenario_list, context_list):
        prompt_template = copy.deepcopy(REWRITE_PT)
        NEW_PB_TAG = "new_problem"
        prompt_list = []
        for context, scenario in zip(context_list, scenario_list):
            num = ori_prompt.count('.') + 1
            prompt = prompt_template.format(
                pb=ori_prompt, scenario=scenario,
                input_variables=context,
                num=num, num_2=num +1,
            )
            prompt_list.append(prompt)
        new_pb_text = self.prompt2text(prompt_list, temperature=0.6, max_tokens=512)
        new_pb = [extract_data(d, NEW_PB_TAG) for d in new_pb_text]
        new_pb = [p[0] for p in new_pb if len(p) > 0]
        return new_pb

    @staticmethod
    def make_new_code_task(ori_code_task: CodeTask, new_instructions):
        new_tasks = []
        TAB = "    "
        for new_instruction in new_instructions:
            new_task = copy.deepcopy(ori_code_task)
            new_instruction = new_instruction.strip()
            new_docstring = f'\n{TAB}{new_instruction}\n{TAB}\n{ori_code_task.demo_str}\n{TAB}'
            new_task.prefix = replace_function_docstring(
                new_task.prefix, new_task.entry_func, new_docstring
            )
            new_task.solution = replace_function_docstring(
                new_task.solution, new_task.entry_func, new_docstring
            )
            new_task.instruction = new_instruction
            new_task.docstring = new_docstring
            new_tasks.append(new_task)
        return new_tasks

    def verifier(self, save_dir, new_save_dir):
        prompt_template = copy.deepcopy(VERIFY_PT)

        for file_name in os.listdir(save_dir):
            data = torch.load(os.path.join(save_dir, file_name), weights_only=False)
            ori_code_task = data[0]
            new_instructions = data[1]

            query_pt = []
            for new_instruction in new_instructions:
                pt = prompt_template.format(
                    inst_a=ori_code_task.instruction, inst_b=new_instruction
                )
                query_pt.append(pt)
            res = self.prompt2text(query_pt, temperature=0.2, max_tokens=10)
            filted_index = [iii for iii, d in enumerate(res) if str(d).lower().startswith('yes')]

            new_instructions = [new_instructions[index] for index in filted_index]
            new_code_tasks = self.make_new_code_task(ori_code_task, new_instructions)
            save_path = os.path.join(new_save_dir, file_name)
            print(ori_code_task['prefix'])
            print('******************************')
            for code_task in new_code_tasks:
                print(code_task['prefix'])
            print('-=--=-------------------------')
            torch.save(new_code_tasks, save_path)

    def _gen(self, data, scenario_dict):

        input_vars = [eval(d.input_str) for d in data.test_cases]

        solution_func_name = data.entry_func
        input_names = extract_function_variables(
            data.solution, solution_func_name)

        func_input_types = infer_detailed_types(input_vars)

        instruction = data.instruction
        scenario_list, context_list = self.context_proposer(
            instruction, input_names, func_input_types, scenario_dict)

        new_pb_list = self.prompt_rewrite(
            ori_prompt=instruction,
            scenario_list=scenario_list,
            context_list=context_list
        )
        save_dir = os.path.join(self.work_dir, data.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, str(data.data_id).replace('/', '_') + '.tar')
        torch.save([data, new_pb_list, scenario_list, context_list], save_path)

    def run_gen(self, seed_dataset):
        with open(self.scenario_path, 'r') as f:
            scenario_dict = json.load(f)

        for data in tqdm(seed_dataset, desc="Processing dataset"):
            self._gen(data, scenario_dict)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--llm_id', type=int, default=0)
#     args = parser.parse_args()
#
#     model_info = AGENT_LIST[args.llm_id]
#     gen = PromptGeneration(
#         model_info=model_info,
#         work_dir=NEW_PROMPT_DIR)
#     gen.scenario_proposer()
#
#     for data_id in range(2):
#         dataset = load_my_dataset(data_id)
#
#         gen.run_gen(dataset)
#
#         ori_save_dir = os.path.join(gen.work_dir, dataset.to_list()[0]['dataset_name'])
#         new_save_dir = os.path.join(gen.work_dir, "filted_" + dataset.to_list()[0]['dataset_name'])
#         os.makedirs(new_save_dir, exist_ok=True)
#         gen.verifier(ori_save_dir, new_save_dir)
