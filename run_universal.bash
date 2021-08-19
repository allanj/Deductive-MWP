#!/bin/bash




diff_param=0
filtered_steps=-1
model_folder=universal_diff_${diff_param}

TOKENIZERS_PARALLELISM=false \
python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} \
                          --diff_param_for_height=${diff_param} --fp16=1 --filtered_steps ${filtered_steps} > logs/${model_folder}.log 2>&1

