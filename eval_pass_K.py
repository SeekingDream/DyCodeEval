import json

import numpy as np

import argparse


import os
import time
import shutil
import tempfile
import subprocess
from multiprocessing import Pool, Queue

from utils import model_id2name_cls
from utils import load_my_dataset, EXE_ENV_DIR
from utils import PARTIAL_LIST, load_lora, make_task_name
from utils import GENERATED_CODE_DIR, SPLIT_SYM, PASS_AT_K_DIR, EXE_RES_DIR


THREAD_NUM = 256
PY_BIN = "/local/arise/CM/miniconda3/envs/codellm/bin/python"

def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))



def execute_script(args):
    file_path, subdir_name = args
    try:
        # Run script inside exec_dir
        result = subprocess.run(
            [PY_BIN, os.path.abspath(file_path)],
            capture_output=True,
            text=True,
            timeout=3,  # Timeout to prevent infinite loops
            cwd=EXE_ENV_DIR  # Set execution directory

        )
        return file_path, subdir_name, result.returncode == 0
    except Exception as e:
        print(f"Error executing {file_path}: {e}")
        return file_path, subdir_name, False


def exec_code(task_dir):
    t1 = time.time()
    execution_results = {}

    tasks = []
    for root, dirs, files in os.walk(task_dir):
        subdir_name = os.path.basename(root)
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                tasks.append((file_path, subdir_name))
    # for task in tasks:
    #     file_path, subdir_name, success = execute_script(task)
    #     file_name = os.path.basename(file_path)
    #     if file_name not in execution_results:
    #         execution_results[file_name] = {}
    #     execution_results[file_name][subdir_name] = success

    # Use multiprocessing Pool for parallel execution
    results_queue = Queue()
    with Pool(processes=THREAD_NUM) as pool:
        async_result = pool.map_async(execute_script, tasks)

        # Collect results from the async result
        for result in async_result.get():
            results_queue.put(result)

    # Organize results into the dictionary format
    while not results_queue.empty():
        file_path, subdir_name, success = results_queue.get()
        file_name = os.path.basename(file_path)
        if file_name not in execution_results:
            execution_results[file_name] = {}
        execution_results[file_name][subdir_name] = success

    t2 = time.time()
    print(f"Execution time: {t2 - t1} seconds")
    return execution_results

def eval_one_dir(eval_dir):
    model_name, data_name, hyper_params = eval_dir.split('/')[-3:]
    task_name = f"{model_name}{SPLIT_SYM}{data_name}{SPLIT_SYM}{hyper_params}"
    exe_res_path = os.path.join(EXE_RES_DIR, f"{task_name}.json")
    pass_k_path = os.path.join(PASS_AT_K_DIR, f"{task_name}.json")

    execution_results = exec_code(eval_dir)

    pass_k = {}
    for name, data in execution_results.items():
        n = len(data)
        c = len([d for d in data.values() if d is True])
        pass_k[name] = {
            "pass@1": estimator(n, c, 1),
            "pass@3": estimator(n, c, 3),
            "pass@5": estimator(n, c, 5),
            "pass@10": estimator(n, c, 10),

            "n": n}
        # Save results to JSON

    with open(exe_res_path, "w") as json_file:
        json.dump(execution_results, json_file, indent=4)

    with open(pass_k_path, "w") as json_file:
        json.dump(pass_k, json_file, indent=4)

    pass_1 = np.mean([r["pass@1"] for r in pass_k.values()])
    pass_3 = np.mean([r["pass@3"] for r in pass_k.values()])
    pass_5 = np.mean([r["pass@5"] for r in pass_k.values()])
    pass_10 = np.mean([r["pass@10"] for r in pass_k.values()])


    return {
        "pass_1": pass_1,
        "pass_3": pass_3,
        "pass_5": pass_5,
        "pass_10": pass_10,

    }

def eval_task(task_dir):
    all_res = {}
    for hyper_name in os.listdir(task_dir):
        eval_dir = os.path.join(task_dir, hyper_name)
        pass_k = eval_one_dir(eval_dir)
        print(task_dir, hyper_name)
        for k in pass_k.keys():
            print(k, pass_k[k])
        all_res[hyper_name] = pass_k
    return all_res


def main(args):

    lora_dataset = load_my_dataset(args.lora_data_id)
    _, _, _, _, model_name = model_id2name_cls(args.model_id)

    partial = PARTIAL_LIST[args.partial_id]
    lora_path = load_lora(model_name, lora_dataset, partial)
    if lora_path is not None:
        task_name = make_task_name(model_name, lora_dataset, partial)
    else:
        task_name = make_task_name(model_name, None, None)
    print(f"model name is {task_name}")

    eval_dataset = load_my_dataset(args.data_id)
    task_dir = os.path.join(
        GENERATED_CODE_DIR, task_name, eval_dataset.data_name,
    )
    eval_task(task_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=10)
    parser.add_argument('--lora_data_id', type=int, default=1)
    parser.add_argument('--partial_id', type=int, default=0)
    parser.add_argument('--data_id', type=int, default=0)
    args = parser.parse_args()
    for model_id in range(19, 27):
        args.model_id = model_id
        for data_id in [0, 1, 2, 3, 4, 5]:
            args.data_id = data_id
            main(args)
