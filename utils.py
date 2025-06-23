import os
from datasets import load_dataset
from datasets import Dataset
from collections import defaultdict
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

from pt import CODE_GEN_PT_DICT

from src.data import CodeGenDataset, HumanEvalZero_Data, MBPP_Data
from src.data import SynCodeGen_Data


from src.codellm import LocalVLLM, AbstLiteLLM, AbstLLM

SPLIT_SYM = "::::"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OVERFIT_DIR = os.path.join(RESULTS_DIR, "overfit_dir")
os.makedirs(OVERFIT_DIR, exist_ok=True)
GENERATED_CODE_DIR = os.path.join(RESULTS_DIR, "generated_code")
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
FINAL_RES = "final_res"
os.makedirs(FINAL_RES, exist_ok=True)
EXE_ENV_DIR = "exe_env"
os.makedirs(EXE_ENV_DIR, exist_ok=True)
NEW_PROMPT_DIR = os.path.join(RESULTS_DIR, "new_prompt")
os.makedirs(NEW_PROMPT_DIR, exist_ok=True)
EXE_RES_DIR = os.path.join(RESULTS_DIR, "exe_res_dir")
os.makedirs(EXE_RES_DIR, exist_ok=True)
PASS_AT_K_DIR = os.path.join(RESULTS_DIR, "pass_at_k")
os.makedirs(PASS_AT_K_DIR, exist_ok=True)

PARTIAL_LIST = [
    0, 0.25, 0.5, 0.75, 1.0
]

STOP_TOKENS_DICT = {
    "chat": [],
    "competition": [
        "\n>>>", "\n$", '\nclass',
        '\ndef', '\n#', '\nprint',
         "\n@", "\nassert", '\nfor',
        "\nif __name__ == '__main__':",
        '\nif __name__ == "__main__":'
    ]
}

def model_id2name_cls(model_id: int):
    API_NUN = 4
    llm_config = {
        "tp_size": 1,
        "max_model_len": 8192,
        "is_lora": False,
        "model_type": "chat"
    }

    if model_id == 0:
        model_name = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        model_cls = AbstLiteLLM
        provider = "bedrock"
    elif model_id == 1:
        model_name = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        model_cls = AbstLiteLLM
        provider = "bedrock"
    elif model_id == 2:
        model_name = "gemini-1.5-flash-002"
        model_cls = AbstLiteLLM
        provider = "vertex_ai"
    elif model_id == 3:
        model_name = "gemini-2.0-flash-001"
        model_cls = AbstLiteLLM
        provider = "vertex_ai"


    elif model_id == API_NUN + 0:
        model_name = "meta-llama/Llama-3.2-1B"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["is_lora"] = True
        llm_config["model_type"] = "competition"
        llm_config['tp_size'] = 8
    elif model_id == API_NUN + 1:
        model_name = "meta-llama/Llama-3.2-3B"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"
        llm_config["is_lora"] = True
    elif model_id == API_NUN + 2:
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["is_lora"] = True
    elif model_id == API_NUN + 3:
        model_name = "meta-llama/Llama-3.1-8B"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"
    elif model_id == API_NUN + 4:
        model_name = "meta-llama/CodeLlama-7b-hf"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"
    elif model_id == API_NUN + 5:
        model_name = "meta-llama/CodeLlama-13b-hf"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"

    elif model_id == API_NUN + 6:
        model_name = "deepseek-ai/DeepSeek-V2-Lite"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"

    elif model_id == API_NUN + 7:
        model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"

    elif model_id == API_NUN + 8:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model_cls = LocalVLLM
        provider = "openai"

    elif model_id == API_NUN + 9:
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        model_cls = LocalVLLM
        provider = "openai"

    elif model_id == API_NUN + 10:
        model_name = "Qwen/Qwen2.5-Coder-7B"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"

    elif model_id == API_NUN + 11:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model_cls = LocalVLLM
        provider = "openai"

    elif model_id == API_NUN + 12:
        model_name = "Qwen/Qwen2.5-7B"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"

    elif model_id == API_NUN + 13:
        model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config['tp_size'] = 8
        llm_config['dtype'] = 'bfloat16'

    elif model_id == API_NUN + 14:
        model_name = "meta-llama/Llama-4-Scout-17B-16E"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config['tp_size'] = 8
        llm_config['dtype'] = 'bfloat16'
        llm_config["model_type"] = "competition"

    elif model_id == API_NUN + 15:
        model_name = "Qwen/Qwen3-0.6B"
        model_cls = LocalVLLM
        provider = "openai"
    elif model_id == API_NUN + 16:
        model_name = "Qwen/Qwen3-0.6B-Base"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"
    elif model_id == API_NUN + 17:
        model_name = "Qwen/Qwen3-8B"
        model_cls = LocalVLLM
        provider = "openai"
    elif model_id == API_NUN + 18:
        model_name = "Qwen/Qwen3-8B-Base"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"
    elif model_id == API_NUN + 19:
        model_name = "Qwen/Qwen3-14B"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config['tp_size'] = 4
    elif model_id == API_NUN + 20:
        model_name = "Qwen/Qwen3-14B-Base"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config["model_type"] = "competition"
        llm_config['tp_size'] = 4
    elif model_id == API_NUN + 21:
        model_name = "Qwen/Qwen3-30B-A3B"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config['tp_size'] = 4
    elif model_id == API_NUN + 22:
        model_name = "Qwen/Qwen3-30B-A3B-Base"
        model_cls = LocalVLLM
        provider = "openai"
        llm_config['tp_size'] = 4
        llm_config["model_type"] = "competition"
    else:
        raise ValueError(f"Model ID {model_id} is not valid")
    shrink_model_name = model_name.split('/')[-1]
    return provider, model_name, model_cls, llm_config, shrink_model_name


def load_finetune_model(model_id: int):
    if model_id in [0, 1, 2]:
        _, model_name, _, _ = model_id2name_cls(model_id)
    else:
        raise ValueError(f"Model {model_id} not supported")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.model_name = model_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_benchmark_model(model_id: int) -> AbstLLM:
    provider, model_name, model_cls, llm_config, shrink_model_name = model_id2name_cls(model_id)
    model = model_cls(provider, model_name, llm_config)

    model.model_name = shrink_model_name
    return model


def make_task_name(model_name, dataset, partial):
    if partial is not None:
        task_name = model_name + SPLIT_SYM + dataset.data_name + SPLIT_SYM + str(partial)
    else:
        assert dataset is None
        task_name = model_name
    return task_name


def load_unique_dataset(dataset_name, split, group_name='data_id', random_seed=33):
    # 1. Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    data_id_to_samples = defaultdict(list)
    for i, sample in enumerate(dataset):
        data_id_to_samples[sample[group_name]].append(sample)

    # 3. For each data_id, randomly select ONE sample using random_seed
    rng = random.Random(random_seed)
    unique_samples = []
    for data_id, samples in data_id_to_samples.items():
        selected = rng.choice(samples)
        unique_samples.append(selected)

    # 4. Optionally, return as a Hugging Face Dataset object
    return Dataset.from_list(unique_samples)

def load_my_dataset(data_id: int, random_seed=33):
    group_name = 'data_id'
    if data_id == 0:
        ds = load_dataset("HeyixInn0/Reorganized-humaneval", split="train")
        data_name = 'HumanEvalZero'
        dataset = HumanEvalZero_Data(ds, data_name)
    elif data_id == 1:
        ds = load_dataset("HeyixInn0/Reorganized-mbpp", split="train")
        data_name = 'MBPP_sanitized'
        dataset = MBPP_Data(ds, data_name)
    elif data_id == 2:
        raw_data_name = "CM/Dynamic_HumanEvalZero"
        split = "Claude3.5_Sonnet"
        ds = load_unique_dataset(raw_data_name, split, group_name, random_seed=random_seed)
        data_name = "Sonnet_HumanEvalZero"
        dataset = HumanEvalZero_Data(ds, data_name)
    elif data_id == 3:
        raw_data_name = "CM/Dynamic_MBPP_sanitized"
        split = "Claude3.5_Sonnet"
        ds = load_unique_dataset(raw_data_name, split, group_name, random_seed=random_seed)
        data_name = "Sonnet_MBPP_sanitized"
        dataset = MBPP_Data(ds, data_name)

    elif data_id == 4:
        raw_data_name = "CM/Dynamic_HumanEvalZero"
        split = "Claude3.5_Haiku"
        ds = load_unique_dataset(raw_data_name, split, group_name, random_seed=random_seed)
        data_name = "Haiku_HumanEvalZero"
        dataset = HumanEvalZero_Data(ds, data_name)

    elif data_id == 5:
        raw_data_name = "CM/Dynamic_MBPP_sanitized"
        split = "Claude3.5_Haiku"
        ds = load_unique_dataset(raw_data_name, split, group_name, random_seed=random_seed)
        data_name = "Haiku_HumanEvalZero"
        dataset = MBPP_Data(ds, data_name)
    else:
        raise NotImplementedError
    return dataset


def load_lora(model_name, dataset, partial):
    task_name = make_task_name(model_name, dataset, partial)
    output_dir = os.path.join(OVERFIT_DIR, task_name)
    if not os.path.exists(output_dir):
        return None
    sub_dirs = sorted(list(os.listdir(output_dir)))
    return os.path.join(output_dir, sub_dirs[-1])


def dataset2ptdict(dataset):
    if isinstance(dataset, CodeGenDataset):
        return CODE_GEN_PT_DICT
    else:
        raise NotImplementedError


if __name__ == '__main__':
    for data_id in range(6):
        data = load_my_dataset(data_id)
    print()
