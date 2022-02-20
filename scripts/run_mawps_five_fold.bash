#!/bin/bash




use_constant=1
add_replacement=1
consider_multiple_m0=1
# use just 'gru' and 'roberta-base' for our best model
var_update_modes=(gru attn)
bert_model_names=(bert-base-cased roberta-base bert-base-multilingual-cased xlm-roberta-base)

cuda_devices=(0 1 3 4 5)
folds=(0 1 2 3 4)

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        for (( f=0; f<${#folds[@]}; f++  )) do
          dev=${cuda_devices[$e]}
          fold_num=${folds[$f]}
          model_folder=mawps_${bert_model_name}_${var_update_mode}_${fold_num}
          python3 universal_main.py \
                  --device=cuda:${dev} \
                  --model_folder=${model_folder} \
                  --mode=train \
                  --height=5 \
                  --num_epochs=100 \
                  --consider_multiple_m0=${consider_multiple_m0} \
                  --train_file=data/mawps-single-five-fold/train_${fold_num}.json \
                  --dev_file=data/mawps-single-five-fold/test_${fold_num}.json \
                  --test_file=none \
                  --add_replacement=${add_replacement} \
                  --train_num=-1 \
                  --dev_num=-1 \
                  --var_update_mode=${var_update_mode} \
                  --bert_model_name=${bert_model_name} \
                  --use_constant=${use_constant} \
                  --fp16=1 \
                  --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
        done
    done
done



