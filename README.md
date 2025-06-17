# `📈 DyCodeEval`

This repository contains the main implementation of **DyCodeEval**, introduced in our ICML 2025 paper:
*“DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination.”*

[🏆 Leaderboard](https://your-leaderboard-link) • [💻 Code](https://github.com/your-username/DyCodeEval) • [🤗 Hugging Face Dataset](https://huggingface.co/datasets/your-dataset) • [🔮 Code Kaleidoscope Project](https://github.com/your-username/DyCodeEval/tree/main/kaleidoscope)


## Introducation 

**DyCodeEval** proposes a novel **dynamic benchmarking framework (dynamic evaluation dataset + dynamic metric)** for evaluating code large language models (Code LLMs). It leverages a **multi-agent cooperation** strategy to **rewrite existing benchmarks** at evaluation time, producing programming problems that are: 1. Semantically equivalent, 2. Diverse, and 3. Non-deterministic. This dynamic generation process helps **mitigate data contamination** and provides a more robust and faithful assessment of a model's reasoning capabilities.

### Design Oveerview
<div  align="center">    
 <img src="https://github.com/SeekingDream/DyCodeEval/blob/main/resource/dycodeeval_overview.jpg" width="760" height="310" alt="Design Overview"/><br/>
</div>   

### 🔧 This repository provides:

* Core implementation of DyCodeEval to dynamically rewrite existing programming problems
* Scripts to reproduce all experiments and benchmarks presented in the paper
* Pre-generated benchmark variants for **HumanEval** and **MBPP**

##  🤗 Pre-generated problems

We provide our pre-generated dataset from HumanEval and MBPP on [HuggingFace](https://huggingface.co/datasets/your-dataset).

## How to Run

### Installation

Install the necessary libraries through `pip install requirement.txt`

### Setup commerical LLM account

We are using `litellm` as our unfilled framework to invoke each LLM, so first follow the [documents](https://github.com/BerriAI/litellm?tab=readme-ov-file#supported-providers-docs) to setup each commerical LLM account.

### Generating Dynamic Benchmark Problems

To generate new benchmark problem, run `python gen_problem.py --data_id=0`.








