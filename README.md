# 📈 DyCodeEval

This repository contains the main implementation of **DyCodeEval**, introduced in our ICML 2025 paper:
*“DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination.”*

[🏆 Leaderboard](https://your-leaderboard-link) • [💻 Code](https://github.com/SeekingDream/DyCodeEval) • [🤗 Hugging Face Dataset](https://huggingface.co/collections/CM/dycodeeval-6858e931f4f1a0d4a29ec2e9) • [🔮 Code Kaleidoscope Project](https://github.com/your-username/DyCodeEval/tree/main/kaleidoscope)


## Introducation 

**DyCodeEval** proposes a novel **dynamic benchmarking framework (dynamic evaluation dataset + dynamic metric)** for evaluating code large language models (Code LLMs). It leverages a **multi-agent cooperation** strategy to **rewrite existing benchmarks** at evaluation time, producing programming problems that are: 1. Semantically equivalent, 2. Diverse, and 3. Non-deterministic. This dynamic generation process helps **mitigate data contamination** and provides a more robust and faithful assessment of a model's reasoning capabilities.

### Design Overview
<div  align="center">    
 <img src="https://github.com/SeekingDream/DyCodeEval/blob/main/resource/dycodeeval_overview.jpg" width="560" height="220" alt="Design Overview"/><br/>
</div>   

### 🔧 This repository provides:

* Core implementation of DyCodeEval to dynamically rewrite existing programming problems
* Scripts to reproduce all experiments and benchmarks presented in the paper
* Pre-generated benchmark variants for **HumanEval** and **MBPP**

### Future Features
* **Support for Fine-Tuning Open Source Models**
  Enable dynamic benchmark generation by fine-tuning open-source code models, removing the dependency on the Claude API.
* **Simplified DyPass\@K Evaluation Script**
  Provide an easy-to-use, unified script for computing `DyPass@K`, replacing the current set of fragmented scripts.


##  🤗 Pre-generated problems


We provide pre-generated HumanEval and MBPP datasets on [Hugging Face](https://huggingface.co/collections/CM/dycodeeval-6858e931f4f1a0d4a29ec2e9).
You can load a dataset using the following code. Here, `raw_data_name` specifies the Hugging Face dataset path, `split` selects either the Sonnet or Haiku pre-generated set, and `random_seed` controls the randomness of the selection.

```python
from utils import load_unique_dataset

raw_data_name = "CM/Dynamic_HumanEvalZero"
split = "Claude3.5_Sonnet"  # or "Claude3.5_Haiku"
dataset = load_unique_dataset(raw_data_name, split, group_name, random_seed=random_seed)
```


## How to Run

### Installation

Install the necessary libraries through `pip install requirement.txt`

### Setup commerical LLM account

We are using `litellm` as our unfilled framework to invoke each LLM, so first follow the [documents](https://github.com/BerriAI/litellm?tab=readme-ov-file#supported-providers-docs) to setup each commerical LLM account.

### Generating Dynamic Benchmark Problems


To generate new benchmark problems use LLM agent, run:

```bash
python gen_problem.py --agent_id=0 --seed_data_id=0 --scenario_num=10 --context_num=10
```
* `agent_id`: Specifies the LLM agent for generation (`0` for Claude 3.5 Haiku, `1` for Claude 3.5 Sonnet).
* `seed_data_id`: Selects the seed dataset (`0` for HumanEval, `1` for MBPP).
* `scenario_num` and `context_num`: Control the number of new data samples generated.


## Dynamic Metric (DyPass@K)


To compute the dynamic metric `DyPass@K`, first run `python gen_problem.py` to generate multiple versions of the benchmark problems. Next, run `python gen_code.py` to generate code for each problem. Finally, execute `python eval_pass_K.py` to compute the `DyPass@K` metric.

We will provide more streamlined, user-friendly scripts for this process in the near future.



## File Structure

* **src/**: Core implementation of the framework

  * **src/code\_llm/**: LLM inference framework, including abstractions for both vLLM and LiteLLM (supporting open-source and commercial code LLMs).
  * **src/data/**: Data structures for various code-related tasks.
  * **src/pt\_opt/**: Implementations of different prompt templates.
  * **src/task\_mutation/**: Algorithms for generating new programming problems.

* **eval\_pass\_K.py**: Computes pass\@K.

* **gen\_code.py**: Generates code for different benchmarks.

* **gen\_problem.py**: Creates new benchmark problems with DyCodeEval.

* **make\_dataset.py**: Prepares datasets and pushes them to Hugging Face.

* **overfit\_train.py**: Overfits code LLMs to simulate data contamination.

* **pt.py**: Prompt templates for code generation.

* **utils.py**: Utility functions.


