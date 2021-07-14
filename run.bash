#!/bin/bash


four_variables=0
use_binary=1
model_folder=math_solver_fv_${four_variables}_bin_${use_binary}

TOKENIZERS_PARALLELISM=false \
python3 main.py --device=cuda:0 --model_folder=${model_folder} --use_binary=${use_binary} \
               --four_variables=${four_variables} --fp16=1 > logs/${model_folder}.log 2>&1

