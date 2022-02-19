#!/bin/bash


model_folder=math23k_baseline

train_file=data/math23k/train23k_processed.json
dev_file=data/math23k/test23k_processed.json
## generation task
#TOKENIZERS_PARALLELISM=false \
#python3 gen_main.py --device=cuda:0 --model_folder=${model_folder} --fp16=1 > logs/${model_folder}.log 2>&1


TOKENIZERS_PARALLELISM=false \
python3 -m seq2seq_baseline --device=cuda:0  --model_folder=${model_folder} \
                     --train_file=${train_file} --dev_file=${dev_file} \
                     --fp16=1 > logs/${model_folder}.log 2>&1

## for generate m0, 1.0, 1.1, 1.2 are length penalty
#TOKENIZERS_PARALLELISM=false python3 -m preprocess.process_four_variables 1.0 ${model_folder} > logs/gen_1.0.log 2>&1
#TOKENIZERS_PARALLELISM=false python3 -m preprocess.process_four_variables 1.1 ${model_folder} > logs/gen_1.1.log 2>&1
#TOKENIZERS_PARALLELISM=false python3 -m preprocess.process_four_variables 1.2 ${model_folder} > logs/gen_1.2.log 2>&1

