#!/bin/bash





model_folder=math23k_parallel

TOKENIZERS_PARALLELISM=false \
python3 parallel_main.py --device=cuda:0 --model_folder=${model_folder} --mode=train --height=10 --num_epochs=100 --consider_multiple_m0=1 \
                          --train_file=data/math23k/train23k_parallel_sorted.json --add_replacement=1 --train_num=-1 --dev_num=-1 \
                          --dev_file=data/math23k/test3k_parallel_sorted.json --use_constant=1 --add_new_token=0 \
                          --diff_param_for_height=0 --fp16=1 --filtered_steps -1 > logs/${model_folder}.log 2>&1
