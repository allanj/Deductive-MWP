#!/bin/bash


add_replacement=1
consider_multiple_m0=1
var_update_modes=(gru attn)
bert_folders=(hfl hfl none none)
bert_model_names=(chinese-bert-wwm-ext chinese-roberta-wwm-ext bert-base-multilingual-cased xlm-roberta-base)
cuda_devices=(0 1 2 3)
epoch=500

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        bert_folder=${bert_folders[$e]}
        dev=${cuda_devices[$e]}
        model_folder=math23k_train_test_${bert_model_name}_${var_update_mode}_${epoch}
        echo "Running mawps with bert model $bert_model_name and var update mode $var_update_mode"
        TOKENIZERS_PARALLELISM=false \
        python3 universal_main.py \
                --device=cuda:${dev} \
                --model_folder=${model_folder} \
                --mode=train \
                --height=10 \
                --train_max_height=14 \
                --num_epochs=${epoch} \
                --consider_multiple_m0=${consider_multiple_m0} \
                --train_file=data/math23k_train_test/combined_train23k_processed_nodup.json \
                --dev_file=data/math23k_train_test/test23k_processed_nodup.json \
                --test_file=none \
                --add_replacement=${add_replacement} \
                --train_num=-1 --dev_num=-1 \
                --bert_folder=${bert_folder} \
                --bert_model_name=${bert_model_name} \
                --var_update_mode=${var_update_mode} \
                --fp16=1 \
                --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
    done
done


## testing
#model_folder=math23k_chinese-roberta-wwm-ext_gru
#bert_model_name=chinese-roberta-wwm-ext
#bert_folder=hfl
#var_update_mode=gru
#python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} --mode=test --height=10 --train_max_height=20 --num_epochs=1000 --consider_multiple_m0=${consider_multiple_m0} \
#                          --train_file=data/math23k/train23k_processed_nodup.json --add_replacement=${add_replacement} --train_num=-1 --dev_num=-1 \
#                          --dev_file=data/math23k/test23k_processed_nodup.json --use_constant=${use_constant} --add_new_token=${add_new_token} \
#                          --bert_folder=${bert_folder} --bert_model_name=${bert_model_name} --var_update_mode=${var_update_mode} --parallel=0 \
#                          --diff_param_for_height=${diff_param} --fp16=1 --filtered_steps ${filtered_steps} --learning_rate=2e-5 > logs/${model_folder}_eval.log 2>&1 &



