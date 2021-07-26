#!/bin/bash



model_folder=mtl

TOKENIZERS_PARALLELISM=false \
python3 mtl_main.py --batch_size=6 --device=cuda:0 --model_folder=${model_folder}  --fp16=1 --parallel=1 > logs/${model_folder}.log 2>&1

