# `📈 DyCodeEval`

This repository contains the main implementation of **DyCodeEval**, introduced in our ICML 2025 paper:
*“DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination.”*

**DyCodeEval** proposes a novel **dynamic benchmarking framework** for evaluating code large language models (Code LLMs). It leverages a **multi-agent cooperation** strategy to **rewrite existing benchmarks** at evaluation time, producing programming problems that are: 1. **Semantically equivalent**, 2. **Diverse**, and 3. **Non-deterministic**.

This dynamic generation process helps **mitigate data contamination** and provides a more robust and faithful assessment of a model's reasoning capabilities.

---

### 🔧 This repository provides:

* Core implementation of DyCodeEval to dynamically rewrite existing programming problems
* Scripts to reproduce all experiments and benchmarks presented in the paper
* Pre-generated benchmark variants for **HumanEval** and **MBPP**

---

### 🔍 Research Insight

Our work reveals a critical flaw in existing code LLM evaluations: **static benchmarks are easily compromised by data contamination**, leading to overestimated model performance. DyCodeEval addresses this issue by dynamically generating problem variants at benchmark time, offering a more reliable and contamination-resilient evaluation pipeline.


## Design Oveerview
<div  align="center">    
 <img src="https://anonymous.4open.science/r/DcL-BD-2E60/figs/Dcl-BD.jpg" width="700" height="350" alt="Design Overview"/><br/>
</div>   
