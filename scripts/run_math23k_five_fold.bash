#!/bin/bash


use_constant=1
add_replacement=1
consider_multiple_m0=1
var_update_modes=(gru attn)
bert_folders=(hfl hfl none none)
bert_model_names=(chinese-bert-wwm-ext chinese-roberta-wwm-ext bert-base-multilingual-cased xlm-roberta-base)
cuda_devices=(0 1 2 3 4)
folds=(0 1 2 3 4)
epoch=1000

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        bert_folder=${bert_folders[$e]}
        for (( f=0; f<${#folds[@]}; f++  )) do
          fold_num=${folds[$f]}
          dev=${cuda_devices[$f]}
          model_folder=math23k_${bert_model_name}_${var_update_mode}_${fold_num}_epoch_${epoch}
          TOKENIZERS_PARALLELISM=false \
          python3 universal_main.py --device=cuda:${dev} \
                        --model_folder=${model_folder} \
                        --mode=train \
                        --height=10 \
                        --train_max_height=14 \
                        --num_epochs=${epoch} \
                        --consider_multiple_m0=${consider_multiple_m0} \
                        --train_file=data/math23k_five_fold/train_${fold_num}.json  \
                        --dev_file=data/math23k_five_fold/test_${fold_num}.json \
                        --test_file=none \
                        --use_constant=${use_constant}  \
                        --add_replacement=${add_replacement} \
                        --train_num=-1 \
                        --dev_num=-1 \
                        --bert_folder=${bert_folder} \
                        --bert_model_name=${bert_model_name} \
                        --var_update_mode=${var_update_mode} \
                        --fp16=1 \
                        --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
        done
    done
done




