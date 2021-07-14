#!/bin/bash


model_folder=generation

TOKENIZERS_PARALLELISM=false \
python3 gen_main.py --device=cuda:0 --model_folder=${model_folder} --fp16=1 > logs/${model_folder}.log 2>&1

