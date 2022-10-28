#!/bin/bash


### Run MathQA using distributed data parallel

var_update_modes=(gru)
bert_model_names=(roberta-large)
batch_size_per_device=8

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        model_folder=mathqa_${bert_model_name}_${var_update_mode}_nowamrup
        CUDA_VISIBLE_DEVICES=0,1,2 \
        accelerate launch universal_main_ddp.py --device=cuda:0 \
                            --model_folder=${model_folder} \
                            --mode=train \
                            --height=12 \
                            --train_max_height=12 \
                            --num_epochs=1000 \
                            --train_file=data/gsm8k/gsm8k_main_aligned_train_unique.json \
                            --dev_file=data/gsm8k/gsm8k_main_aligned_test_unique.json \
                            --test_file=none \
                            --batch_size=${batch_size_per_device} \
                            --train_num=-1 \
                            --dev_num=-1  \
                            --var_update_mode=${var_update_mode} \
                            --bert_model_name=${bert_model_name} \
                            --fp16=1  \
                            --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
      done
done



##eval doe
#bert_model_name=roberta-base
#var_update_mode=gru
#model_folder=mathqa_${bert_model_name}_${var_update_mode}
#python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} --mode=test --height=10 --train_max_height=15 --num_epochs=1000 --consider_multiple_m0=${consider_multiple_m0} \
#                  --train_file=data/MathQA/mathqa_train_nodup.json --add_replacement=${add_replacement} --train_num=-1 --dev_num=-1 --batch_size=5 --var_update_mode=${var_update_mode} \
#                  --dev_file=data/MathQA/mathqa_test_nodup.json --bert_model_name=${bert_model_name} --use_constant=${use_constant} --add_new_token=${add_new_token} \
#                  --diff_param_for_height=${diff_param} --fp16=1 --filtered_steps ${filtered_steps} --parallel=1 --learning_rate=2e-5 > logs/${model_folder}.log 2>&1 &