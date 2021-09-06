#!/bin/bash




diff_param=0
filtered_steps=-1
use_constant=1
add_replacement=1
model_folder=math23k_val_acc_replace_80

TOKENIZERS_PARALLELISM=false \
python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} --height=6 --num_epochs=80 \
                          --train_file=data/math23k/train23k_processed_replacement.json --add_replacement=${add_replacement} \
                          --dev_file=data/math23k/test23k_processed_replacement.json --use_constant=${use_constant} \
                          --diff_param_for_height=${diff_param} --fp16=1 --filtered_steps ${filtered_steps} > logs/${model_folder}.log 2>&1

