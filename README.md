# CheckGPT-v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

The official repository of paper: "On the Detectability of ChatGPT Content: Benchmarking, Methodology, and Evaluation through the Lens of Academic Writing".

## Table of Contents

- [Data](#data)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Data
There are tow versions of datasets: 1. GPABenchmark. 2. GPABench2.
> We mainly use GPABench2 in our CCS 2024 submission.

### Understanding the dataset
**GPABenchmark:**
- GPT data example: ./GPABenchmark/CS_Task1/gpt.json *(Computer Science, Task 1 GPT-WRI)*
- HUM data example: ./GPABenchmark/CS_Task1/hum.json
- Data structure: {PaperID} : {Abstract}

**GPABench2:**
- GPT data example: ./GPABench2/PHX/gpt_task3_prompt4.json *(Physics, Task 3 GPT-POL, Prompt 4)*
- HUM data example: ./GPABench2/PHX/ground.json
- Data structure: {Index} : { {"id"}:{PaperID}, {"title"}:{PaperTitle}, {"abstract"}:{Abstract} }

## Features
To turn text into features, use [*features.py*](CheckGPT/features.py).

## Environment Setup
Run
```bash
pip install -r requirements.txt
```

## Usage
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



