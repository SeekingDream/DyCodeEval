import os
import json
import numpy as np
from tqdm import tqdm
import csv
from collections import defaultdict
import time

from utils import PASS_AT_K_DIR, SPLIT_SYM, FINAL_RES

def avg_list_of_dict(list_of_dicts):
    averages = defaultdict(float)
    count = len(list_of_dicts)

    # Sum up values for each key
    for entry in list_of_dicts:
        for key, value in entry.items():
            averages[key] += value

    # Compute the mean for each key
    averages = {key: value / count for key, value in averages.items()}

    return averages

def compute_avg_pass_at_k(pass_k):
    grouped_res = {}
    for data_id in pass_k.keys():
        k = str(data_id).split("____SPLIT____")[0]
        if k not in grouped_res:
            grouped_res[k] = []
        grouped_res[k].append(pass_k[data_id])
    avg_res = {}
    for k in grouped_res:
        avg_res[k] = avg_list_of_dict(grouped_res[k])

    pass_1 = np.mean([r["pass@1"] for r in avg_res.values()])
    pass_3 = np.mean([r["pass@3"] for r in avg_res.values()])
    pass_5 = np.mean([r["pass@5"] for r in avg_res.values()])
    pass_10 = np.mean([r["pass@10"] for r in avg_res.values()])

    return [pass_1, pass_3, pass_5, pass_10]


def extract_meta(file_name):
    file_name_info = file_name.split(SPLIT_SYM)
    if len(file_name_info) == 4:
        model_name, data_name, hyper_name = (
            file_name_info[0],
            file_name_info[1],
            file_name_info[2] + SPLIT_SYM + file_name_info[3])
    elif len(file_name_info) == 6:
        model_name, data_name, hyper_name = (
            file_name_info[0] + SPLIT_SYM + file_name_info[1] + SPLIT_SYM + file_name_info[2],
            file_name_info[3],
            file_name_info[4] + SPLIT_SYM + file_name_info[5])
    else:
        raise ValueError(f'Invalid file name: {file_name}')
    hyper_name = hyper_name.replace('.json', "")
    model_name = model_name.split('/')[-1]
    return model_name, data_name, hyper_name



def save_dict_to_csv(data_dict, metric_name):
    for hyper_name in data_dict:
        save_file = os.path.join(FINAL_RES, f"{hyper_name}_{metric_name}_results.csv")
        # Write to CSV
        rows = data_dict[hyper_name]

        data_names = set(key for model_data in rows.values() for key in model_data.keys())
        names = sorted(data_names)
        with open(save_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            head_names = [[n + ":pass@1", n + ":pass@3", n + ":pass@5", n + ":pass@10"] for n in names]
            head_names = [item for sublist in head_names for item in sublist]
            header = ['Model'] + head_names
            writer.writerow(header)

            for model in sorted(rows.keys()):
                row = [model] + [rows[model].get(data, ['N/A', 'N/A', 'N/A']) for data in names]
                flattened_row = [model] + [item for sublist in row[1:] for item in sublist]  # Flatten pass@k values
                writer.writerow(flattened_row)

        print(f'Saved: {save_file}')


def main():
    pass_k_res = {}
    for file_name in tqdm(os.listdir(PASS_AT_K_DIR)):
        path = os.path.join(PASS_AT_K_DIR, file_name)
        print(path)
        with open(path, 'r') as f:
            data = json.load(f)
        model_name, data_name, hyper_name = extract_meta(path)
        if hyper_name not in pass_k_res:
            pass_k_res[hyper_name] = defaultdict(dict)

        pass_k_res[hyper_name][model_name][data_name] = compute_avg_pass_at_k(data)
    save_dict_to_csv(pass_k_res, 'pass_at_k')


if __name__ == '__main__':
    main()

