import os
import torch
from utils import load_my_dataset
from copy import deepcopy
from datasets import Dataset, load_dataset, DatasetDict


_agent_name_dict = {
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": "Claude3.5_Haiku",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "Claude3.5_Sonnet",
}

_data_dir = "/home/sc5687/2_Project/CameraReady/DyCodeEval/results/new_prompt"

for raw_data_id in [0, 1]:
    seed_dataset = load_my_dataset(raw_data_id)
    seed_data_name = seed_dataset.data_name
    seed_dataset = seed_dataset._ds
    seed_data_dict = {}
    for seed in seed_dataset:
        seed_data_dict[str(seed['data_id'])] = seed

    data_dict = {}
    for _agent_name, model_name in _agent_name_dict.items():
        task_data_dir = os.path.join(
            _data_dir, _agent_name, f"filted_{seed_data_name}")

        new_dataset = []
        for data_name in os.listdir(task_data_dir):
            data_path = os.path.join(task_data_dir, data_name)
            data_list = torch.load(data_path, weights_only=False)
            for i, data in enumerate(data_list):
                try:
                    ori_data = seed_data_dict[data.data_id]
                except:
                    print(data.data_id)
                new_data = deepcopy(ori_data)
                new_data['random_id'] = i
                new_data['prefix'] = data.prefix
                new_data['doc_string'] = data.docstring
                new_data['solution'] = data.solution
                del new_data["type"]
                new_dataset.append(new_data)


        dataset = Dataset.from_list(new_dataset)
        data_dict[model_name] = dataset

    dataset = DatasetDict(data_dict)
    dataset.push_to_hub(f"CM/Dynamic_{seed_data_name}")
    print(f"CM/Dynamic_{seed_data_name} Success")



