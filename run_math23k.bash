#!/bin/bash




diff_param=0
filtered_steps=-1
model_folder=math23k_val_acc

TOKENIZERS_PARALLELISM=false \
python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} --height=6 \
                          --train_file=data/math23k/train23k_processed_labeled.json \
                          --dev_file=data/math23k/test23k_processed_labeled.json \
                          --diff_param_for_height=${diff_param} --fp16=1 --filtered_steps ${filtered_steps} > logs/${model_folder}.log 2>&1

