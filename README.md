# Code for Large Language Models as Optimizers

This is the code for the work "Large Language
Models as Optimizers" (https://arxiv.org/abs/2309.03409).

## Dependencies

The code has been verified to work under Python 3.10.13 with the following dependencies:

- absl-py (2.0.0)
- google.generativeai (0.1.0)
- immutabledict (3.0.0)
- openai (0.27.2)

## Usage

### Prompt optimization 
Use `opro/optimization/optimize_instructions.py`, follow the steps at the top. 

A quickstarter:

`
python optimize_instructions.py --optimizer="gpt-3.5-turbo" --scorer="text-bison"
--instruction_pos="Q_beginning" --dataset="gsm8k" --task="train" --palm_api_key="<your_palm_api_key>" --openai_api_key="<your_openai_api_key>"
`

### Prompt evaluation
Use `opro/evaluation/evaluate_instructions.py`, follow the steps at the top.

A quickstarter:

`
python evaluate_instructions.py --scorer="text-bison" --dataset="gsm8k" --task="test" --instruction_pos="Q_beginning" --evaluate_training_fold=false --evaluate_test_fold=true --palm_api_key="<your_palm_api_key>"
`


*Disclaimer: this is not an officially supported Google product.*
