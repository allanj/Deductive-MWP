#!/bin/bash



var_update_modes=(gru)
bert_folders=(hfl)
bert_model_names=(chinese-roberta-wwm-ext)
epoch=500
batch_size_per_device=15 # for two devices.
port=9899

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        bert_folder=${bert_folders[$e]}
        model_folder=math23k_${bert_model_name}_${var_update_mode}_${epoch}
        echo "Running mawps with bert model $bert_model_name and var update mode $var_update_mode"
        TOKENIZERS_PARALLELISM=false \
        CUDA_VISIBLE_DEVICES=0,1 \
        accelerate launch --main_process_port ${port}  \
                universal_main_ddp.py --device=cuda:0 \
                --model_folder=${model_folder} \
                --mode=train \
                --batch_size=${batch_size_per_device} \
                --height=10 \
                --train_max_height=14 \
                --num_epochs=${epoch} \
                --train_file=data/math23k_train_test/combined_train23k_processed_nodup.json \
                --dev_file=data/math23k_train_test/test23k_processed_nodup.json \
                --test_file=data/math23k_train_test/test23k_processed_nodup.json \
                --train_num=-1 --dev_num=-1 \
                --bert_folder=${bert_folder} \
                --bert_model_name=${bert_model_name} \
                --var_update_mode=${var_update_mode} \
                --fp16=1 \
                --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
    done
done



