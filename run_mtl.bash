#!/bin/bash



model_folder=mtl_m0_info
mode=train

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2 \
python3 mtl_main.py --batch_size=12 --device=cuda:0 --model_folder=${model_folder}  --fp16=1 --parallel=1 \
		     --mode=${mode}  > logs/${model_folder}_${mode}.log 2>&1

