#!/bin/bash




diff_param=0
filtered_steps=-1
use_constant=1
add_replacement=1
consider_multiple_m0=1
add_new_token=1
model_folder=math23k_mm_num_tok

TOKENIZERS_PARALLELISM=false \
python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} --mode=train --height=10 --num_epochs=100 --consider_multiple_m0=${consider_multiple_m0} \
                          --train_file=data/math23k/train23k_processed_all.json --add_replacement=${add_replacement} --train_num=-1 --dev_num=-1 \
                          --dev_file=data/math23k/test23k_processed_all.json --use_constant=${use_constant} --add_new_token=${add_new_token} \
                          --diff_param_for_height=${diff_param} --fp16=0 --filtered_steps ${filtered_steps} > logs/${model_folder}.log 2>&1
