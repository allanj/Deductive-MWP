#!/bin/bash



model_folder=mtl_wo_m0_fix
mode=train

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1 \
python3 mtl_main.py --batch_size=12 --device=cuda:0 --model_folder=${model_folder}  --fp16=1 --parallel=1 \
		     --mode=${mode} --insert_m0_string=0 > logs/${model_folder}_${mode}.log 2>&1

