import os
import argparse
import torch
import json

from copy import deepcopy

from utils import load_benchmark_model
from utils import load_my_dataset, dataset2ptdict
from utils import PARTIAL_LIST, load_lora, make_task_name
from utils import GENERATED_CODE_DIR, SPLIT_SYM, STOP_TOKENS_DICT


def generate(code_llm, pt, eval_dataset, sample_num, save_dir, override):

    for i in range(sample_num):
        task_dir = os.path.join(save_dir, str(i))
        os.makedirs(task_dir, exist_ok=True)

    final_eval_dataset = []
    for data in eval_dataset:
        for i in range(sample_num):
            tmp_data = deepcopy(data)
            file_id = str(tmp_data.data_id).replace('/', '-')
            task_dir = os.path.join(save_dir, str(i))
            save_path = os.path.join(task_dir, f"code_{file_id}.py")

            tmp_data.save_path = save_path
            tmp_data.task_dir = task_dir
            final_eval_dataset.append(tmp_data)

    res = code_llm.gen_batch(pt, final_eval_dataset)
    raw_res_path = os.path.join(save_dir, 'raw_res.tar')
    if override or not os.path.exists(raw_res_path):
        torch.save(res, raw_res_path)
        print(f'save meta data to {raw_res_path}')

    for output in res:
        save_file = output.ori_task.save_path

        import_st = output.ori_task.import_str
        test_cases = output.ori_task.entry_code
        final_code = import_st + '\n\n\n' + output.final_ans + '\n\n\n' + test_cases

        if override or not os.path.exists(save_file):
            with open(save_file, 'w') as f:
                f.write(final_code)


def main(args):
    eval_data_list = [load_my_dataset(data_id) for data_id in range(6, 7)]

    sample_num = 1 if args.temperature == 0 else args.n
    lora_dataset = load_my_dataset(args.lora_data_id)

    code_llm = load_benchmark_model(args.model_id)

    partial = PARTIAL_LIST[args.partial_id]
    lora_path = load_lora(code_llm.model_name, lora_dataset, partial)
    if lora_path is not None:
        task_name = make_task_name(code_llm.model_name, lora_dataset, partial)
    else:
        task_name = make_task_name(code_llm.model_name, None, None)


    print(f"model name is {task_name}")
    config = {
        'temperature':  args.temperature,
        "top_p": args.top_p,
        "max_tokens": 8192,  # args.max_tokens,
        "dtype": "auto",
        'lora_path': lora_path,
        "stop": STOP_TOKENS_DICT[code_llm.model_type],

    }
    code_llm.init_ai_kwargs(config)

    for eval_dataset in eval_data_list:
        pt_dict = dataset2ptdict(eval_dataset)
        pt = pt_dict[code_llm.model_type]
        save_dir = os.path.join(
            GENERATED_CODE_DIR, task_name, eval_dataset.data_name,
            f"temp_{args.temperature}{SPLIT_SYM}top_p{args.top_p}"
        )
        os.makedirs(save_dir, exist_ok=True)
        if args.override or not os.path.join(save_dir, 'config.json'):
            with open(os.path.join(save_dir, 'config.json'), 'w') as f:
                json.dump(config, f)
        generate(code_llm, pt, eval_dataset, sample_num, save_dir, args.override)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--model_id', type=int, default=6)
    parser.add_argument('--lora_data_id', type=int, default=1)
    parser.add_argument('--partial_id', type=int, default=0)
    parser.add_argument('--data_id', type=int, default=6)
    parser.add_argument('--override', type=bool, default=True)

    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()

    main(args)
