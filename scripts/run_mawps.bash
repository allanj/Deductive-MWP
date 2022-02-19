#!/bin/bash




diff_param=0
filtered_steps=-1
use_constant=1
add_replacement=0
consider_multiple_m0=1
add_new_token=0
var_update_modes=(gru attn)
bert_model_names=(bert-base-cased roberta-base bert-base-multilingual-cased xlm-roberta-base)
cuda_devices=(0 1 2 3)

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        dev=${cuda_devices[$e]}
        model_folder=mawps_${bert_model_name}_${var_update_mode}
        echo "Running mawps with bert model $bert_model_name and var update mode $var_update_mode"
        TOKENIZERS_PARALLELISM=false \
        python3 universal_main.py --device=cuda:${dev} --model_folder=${model_folder} --mode=train --height=5 --num_epochs=1000 --consider_multiple_m0=${consider_multiple_m0} \
                          --train_file=data/mawps-single/mawps_train_nodup.json --add_replacement=${add_replacement} --train_num=-1 --dev_num=-1 --var_update_mode=${var_update_mode} \
                          --dev_file=data/mawps-single/mawps_valid_nodup.json --test_file=data/mawps-single/mawps_test_nodup.json --bert_model_name=${bert_model_name} --use_constant=${use_constant} --add_new_token=${add_new_token} \
                          --diff_param_for_height=${diff_param} --fp16=1 --filtered_steps ${filtered_steps} --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
    done
done



