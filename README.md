# CheckGPT-v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

The official repository of paper: "On the Detectability of ChatGPT Content: Benchmarking, Methodology, and Evaluation through the Lens of Academic Writing".

## Table of Contents

- [Data](#data)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)


## Data
There are tow versions of datasets: 1. GPABenchmark. 2. GPABench2.
> We mainly use GPABench2 in our CCS 2024 submission.

### Description of the Datasets
**GPABenchmark:**
- GPT example: ./GPABenchmark/CS_Task1/gpt.json *(Computer Science, Task 1 GPT-WRI)*
- HUM example: ./GPABenchmark/CS_Task1/hum.json
- Data structure: {PaperID} : {Abstract}

**GPABench2:**
- GPT example: ./GPABench2/PHX/gpt_task3_prompt4.json *(Physics, Task 3 GPT-POL, Prompt 4)*
- HUM example: ./GPABench2/PHX/ground.json
- Data structure: {Index} : { {"id"}:{PaperID}, {"title"}:{PaperTitle}, {"abstract"}:{Abstract} }

For HUM Task 2 GPT-CPL, use the second half of each text. See this line of [code]().

### Other Datasets used in this Paper:
- Other Academic Writing Purposes (Section 5.4.1)
- Classic NLP Datasets (Section 5.4.2)
- Advanced Prompt Engineering (Section 5.7)
- Sanitized GPT Output (Section 5.10)

## Features

### Feature Extraction
To turn text into features, use [*features.py*](CheckGPT/features.py).

### Paper Abstract Examples for Quick Test
We also provide some mini sets of testing examples for quick experiments.

## Pretrained Models:
Download via [Google Drive]()

## Environment Setup
Run
```bash
pip install -r requirements.txt
```

## Usage
### Basic Usage
For training, testing and transfer learning, use [*dnn.py*](CheckGPT/dnn.py) using this format:
```bash
python dnn.py {SUBJECT} {TASK} {EXP_ID} 
```

**Examples:**
1. To train a model from scratch on CS and task 1:
```bash
python dnn.py CS_save 1 0001 
```

2. To test a model on HSS and task 3:
```bash
python dnn.py HSS 3 0001 --test 1
```

3. To evaluate any text, run and follow instructions:
```bash
python test.py
```

## Notes
Here we provide the references for other experiments in the paper:
### Benchmarking Online and Open-source ChatGPT Detectors (Section 2.2)
- GPTZero
- ZeroGPT
- OpenAI's classifier
- HC3-PPL
- HC3-GLTR
- HC3-RBT
- OpenAI-RBT
- HLR, Rank, Log-Rank, TP, PPL, Entropy
- DetectGPT
- BERT
- DistillBERT
- RoBERTa
- GPT-2

### SOTA ChatGPT Datasets in the Literature (Section 5.4.3) & non-GPT LLMs (Section 5.6.2)
- ArguGPT
- HC3
- M4
- MULTITuDE
- MGTBench

### CheckGPT Performance Over Time (Section 5.5)
- ChatLog-HC3

