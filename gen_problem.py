import os
from utils import NEW_PROMPT_DIR, load_my_dataset
from utils import load_benchmark_model
from src.task_mutation.dycodeeval import ProblemGen

import argparse



def main(agent_id, seed_data_id, scenario_num, context_num):
    agent = load_benchmark_model(agent_id)
    work_dir = str(os.path.join(NEW_PROMPT_DIR, agent.model_name))
    gen = ProblemGen(
        agent=agent,
        work_dir=work_dir,
        scenario_num=scenario_num,
        context_num=context_num,
    )
    gen.scenario_proposer()
    dataset = load_my_dataset(seed_data_id)

    gen.run_gen(dataset)

    ori_save_dir = os.path.join(gen.work_dir, dataset.data_name)
    new_save_dir = os.path.join(gen.work_dir, "filted_" + dataset.data_name)
    os.makedirs(new_save_dir, exist_ok=True)
    gen.verifier(ori_save_dir, new_save_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_id', type=int, default=0)
    parser.add_argument('--seed_data_id', type=int, default=0)

    parser.add_argument('--scenario_num', type=int, default=10)
    parser.add_argument('--context_num', type=int, default=10)

    args = parser.parse_args()


    main(
        agent_id=args.agent_id,
        seed_data_id=args.seed_data_id,
        scenario_num=args.scenario_num,
        context_num=args.context_num,
    )
