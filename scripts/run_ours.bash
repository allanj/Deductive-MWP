#!/bin/bash




diff_param=0
use_constant=1
add_replacement=1
consider_multiple_m0=1
add_new_token=0
var_update_modes=(gru attn)
bert_folders=(hfl hfl none none)
bert_model_names=(chinese-bert-wwm-ext chinese-roberta-wwm-ext bert-base-multilingual-cased xlm-roberta-base)
cuda_devices=(0 1 2 3)

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        bert_folder=${bert_folders[$e]}
        dev=${cuda_devices[$e]}
        model_folder=math97k_${bert_model_name}_${var_update_mode}
#        echo "Running mawps with bert model $bert_model_name and var update mode $var_update_mode"
        TOKENIZERS_PARALLELISM=false \
        python3 universal_main.py --device=cuda:${dev} --model_folder=${model_folder} --mode=train --height=10 --num_epochs=300 --consider_multiple_m0=${consider_multiple_m0} \
                           --train_file=data/large_math/large_math_train_nodup.json --add_replacement=${add_replacement} --train_num=-1 --dev_num=-1 \
                           --dev_file=data/large_math/large_math_test_nodup.json --use_constant=${use_constant} --add_new_token=${add_new_token} \
                           --bert_folder=${bert_folder} --bert_model_name=${bert_model_name} --parallel=0 --batch_size=24 \
                           --diff_param_for_height=${diff_param} --fp16=1 --learning_rate=2e-5 > logs/${model_folder}.log 2>&1
    done
done




## eval
#CUDA_VISIBLE_DEVICES=0 \
#python3 universal_main.py --device=cuda:0 --model_folder=${model_folder} --mode=test --height=10 --num_epochs=200 --consider_multiple_m0=${consider_multiple_m0} \
#                          --train_file=data/large_math/large_math_train_nodup.json --add_replacement=${add_replacement} --train_num=-1 --dev_num=-1 \
#                          --dev_file=data/large_math/large_math_test_nodup.json --use_constant=${use_constant} --add_new_token=${add_new_token} \
#                          --bert_folder=hfl --bert_model_name=chinese-roberta-wwm-ext --parallel=0 --batch_size=24 \
#                          --diff_param_for_height=${diff_param} --fp16=1 --learning_rate=2e-5 > logs/${model_folder}_eval_beam_3.log 2>&1





