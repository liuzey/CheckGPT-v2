# CheckGPT-v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

The official repository of "On the Detectability of ChatGPT Content: Benchmarking, Methodology, and Evaluation through the Lens of Academic Writing".

## Table of Contents

- [Data](#data)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)


## Data
There are two versions of datasets:
1. GPABenchmark. 
2. GPABench2.
> We mainly use GPABench2 in our CCS 2024 submission.

### Description of the Datasets
**GPABenchmark:**
- GPT example: ./GPABenchmark/CS_Task1/gpt.json *(Computer Science, Task 1 GPT-WRI)*
- HUM example: ./GPABenchmark/CS_Task1/hum.json
- Data structure: {PaperID}: {Abstract}

**GPABench2:**
- GPT example: ./GPABench2/PHX/gpt_task3_prompt4.json *(Physics, Task 3 GPT-POL, Prompt 4)*
- HUM example: ./GPABench2/PHX/ground.json
- Data structure: 
{Index}: 
{ 
{"id"}: {PaperID},
{"title"}: {PaperTitle},
{"abstract"}: {Abstract}
}

For HUM Task 2 GPT-CPL, use the second half of each text.

### Other Datasets used in this Paper:
Download these files and put them under *CheckGPT_presaved_files*:
- Other Academic Writing Purposes (Section 5.4) (Available under *CheckGPT_presaved_files/Additional_data/Other_purpose*)
- Classic NLP Datasets (Section 5.4) (Available under *CheckGPT_presaved_files/Additional_data/Classic_NLP*)
- Advanced Prompt Engineering (Section 5.7) (Available under *CheckGPT_presaved_files/Additional_data/Prompt_engineering*)
- Sanitized GPT Output (Section 5.10) (Available under *CheckGPT_presaved_files/Additional_data/Sanitized*)
- GPT4 (Section 5.6 )  (Available under *CheckGPT_presaved_files/Additional_data/GPT4*)

## Pre-trained Models:
Download. Place them under *CheckGPT_presaved_files*.
- Models trained on GPABenchmark (v1) can be accessed at *Pretrained_models*.
- Experiments in Section 5.2 and 5.3, including pre-trained models and training logs, can be found at *saved_experiments/basic*.

## Environment Setup
Run
```bash
pip install -r requirements.txt
```

## Features
To train or reuse the text, please extract features from the text beforehand (For development only. Not need for testing).
### Feature Extraction
To turn text into features, use [*features.py*](CheckGPT/features.py). 
```bash
python features.py {DOMAIN} {TASK} {PROMPT}
```
Features will be saved in the folder named *embeddings*.
**ATTENTION: Each file of saved features for 50,000 samples will be approximately 52GB.**

For example, to fetch the features of GPT data in **CS** on **Task 1 Prompt 3**:
```bash
python features.py CS 1 3 --gpt 1
```
The saved features are named in this format: *./embeddings/CS/gpt_CS_task1_prompt3.h5*

Likely, to fefetch the features of HUM data in **CS** on **Task 1 Prompt 3**:
```bash
python features.py CS 1 3 --gpt 0
```
The saved features are named in this format: *./embeddings/CS/ground_CS.h5* (Same for Task 1 and 3)

For Task 2 GPT-CPL, the ground data will be cut into halves. Only the second halves will be processed. An example of saved names is *ground_CS_task2.h5*.

Or you can name the desired sample size. For example, to get the first 1000 samples:
```bash
python features.py CS 1 3 --gpt 0 --number 1000
```
The saved features are named in this format: *./embeddings/CS_1000/gpt_CS_task1_prompt3.h5*


## Usage
### On-the-fly
To evaluate any single piece of input text, run and follow instructions:
```bash
python run_input.py
```

### Testing on text files
To directly evaluate any json data file, run:
```bash
python validate_text.py {FILE_PATH} {MODEL_PATH} {IS_GPT_OR_NOT}
```
For example, if you want to test pre-trained model *../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth* on *../GPABench2/CS/gpt_task3_prompt2.json* or *../GPABench2/CS/ground.json*:
```bash
python validate_text.py ../GPABench2/CS/gpt_task3_prompt2.json ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth 1
```
or
```bash
python validate_text.py ../GPABench2/CS/ground.json ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth 0
```

To run it on special dataset like GPT4, run
```bash
python validate_text.py ../CheckGPT_presaved_files/Additional_data/GPT4/chatgpt_cs_task3.json ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth 1
```

### Testing on pre-saved features
```bash
python dnn.py {DOMAIN} {TASK} {PROMPT} {EXP_ID} --pretrain 1 --test 1 --saved-model {MODEL_PATH}
```

To test the pretrained model *../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth* on pre-save features *./embeddings/CS/gpt_task3_prompt2.h5* and *./embeddings/CS/ground.h5*, run
```bash
python dnn.py CS 3 2 12345 --pretrain 1 --test 1 --saved-model ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth
```

For features of small test data with 1000 samples:
```bash
python dnn.py CS_1000 3 2 12346 --pretrain 1 --test 1 --saved-model ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth
```

### Training on pre-saved features
```bash
python dnn.py {DOMAIN} {TASK} {PROMPT} {EXP_ID}
```

To train a model from scratch on CS Task 3 Prompt 2:
```bash
python dnn.py CS 3 2 12347
```

**Ablation Study:** use --modelid to use different model (0 for CheckGPT, 1 for RCH, 2 for MLP-Pool, 3 for CNN):
```bash
python dnn.py CS 3 2 12347 --modelid 1
python dnn.py CS 3 2 12347 --modelid 2
python dnn.py CS 3 2 12347 --modelid 3
```

### Transfer Learning
```bash
python dnn.py {DOMAIN} {TASK} {PROMPT} {EXP_ID} --trans 1 --mdomain --mtask --mprompt --mid
```
At the beginning, it will also provide cross-validation (testing) result.

For example, to transfer from CS_Task3_Prompt1 to HSS_Task1_Prompt2, run:
```bash
python dnn.py HSS 1 2 12347 --trans 1 --mdomain CS --mtask 3 --mprompt 1 --mid 12346
python dnn.py HSS_500 1 2 12347 --trans 1 --mdomain CS_500 --mtask 3 --mprompt 1 --mid 12346
```
--mid indicates the pre-trained model in previous experiments (e.g., 12346 as we did above).

