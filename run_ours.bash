#!/bin/bash




diff_param=0
use_constant=1
add_replacement=1
consider_multiple_m0=1
add_new_token=0
model_folder=ours

TOKENIZERS_PARALLELISM=false \
python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} --mode=train --height=10 --num_epochs=200 --consider_multiple_m0=${consider_multiple_m0} \
                          --train_file=data/large_math/large_math_train_nodup.json --add_replacement=${add_replacement} --train_num=-1 --dev_num=-1 \
                          --dev_file=data/large_math/large_math_test_nodup.json --use_constant=${use_constant} --add_new_token=${add_new_token} \
                          --bert_folder=hfl --bert_model_name=chinese-roberta-wwm-ext \
                          --diff_param_for_height=${diff_param} --fp16=1 --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
