import traceback

import torch
from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, BertTokenizer, RobertaTokenizer, RobertaTokenizerFast
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import numpy as np
from src.utils import read_data, write_data
import collections
import re
from src.eval.utils import compute_value_for_incremental_equations
import math
from typing import Dict, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

## "<quant>" token will be split into different subwords in different tokenizers
class_name_2_quant_list = {
    "bert-base-cased": ['<', 'q', '##uant', '>'],
    "roberta-base": ['Ġ<', 'quant', '>'],
    "coref-roberta-base": ['Ġ<', 'quant', '>'],
    "roberta-large": ['Ġ<', 'quant', '>'],
    "bert-base-multilingual-cased": ['<', 'quant', '>'],
    "xlm-roberta-base": ['▁<', 'quant', '>'],
    'bert-base-chinese': ['<', 'q', '##uan', '##t', '>'],
    'hfl/chinese-bert-wwm-ext': ['<', 'q', '##uan', '##t', '>'],
    'hfl/chinese-roberta-wwm-ext': ['<', 'q', '##uan', '##t', '>'],
}

UniFeature = collections.namedtuple('UniFeature', 'input_ids attention_mask token_type_ids variable_indexs_start variable_indexs_end num_variables variable_index_mask labels label_height_mask')
UniFeature.__new__.__defaults__ = (None,) * 7

class UniversalDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 uni_labels:List[str],
                 pretrained_model_name:str,
                 number: int = -1,
                 filtered_steps: List = None,
                 constant2id: Dict[str, int] = None,
                 constant_values: List[float] = None,
                 data_max_height: int = 100) -> None:
        self.tokenizer = tokenizer
        self.constant2id = constant2id
        self.constant_values = constant_values
        self.constant_num = len(self.constant2id) if self.constant2id else 0
        self.data_max_height = data_max_height
        self.uni_labels = uni_labels
        self.quant_list = class_name_2_quant_list[pretrained_model_name]
        filtered_steps = [int(v) for v in filtered_steps] if filtered_steps is not None else None
        self.read_math23k_file(file, tokenizer, number, filtered_steps)


    def read_math23k_file(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1,
                 filtered_steps: List = None) -> None:
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        max_num_steps = 0
        self.insts = []
        num_step_count = Counter()
        equation_layer_num = 0
        equation_layer_num_count = Counter()
        var_num_all =0
        var_num_count = Counter()
        sent_len_all = 0
        filter_type_count = Counter()
        found_duplication_inst_num = 0
        filter_step_count = 0
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            mapped_text = obj["text"]
            sent_len = len(mapped_text.split())
            ## replace the variable with <quant>
            for k in range(ord('a'), ord('a') + 26):
                mapped_text = mapped_text.replace(f"temp_{chr(k)}", " <quant> ")
            ## obtain the text string
            if "math23k" in file:
                mapped_text = mapped_text.split()
                input_text = ""
                for idx, word in enumerate(mapped_text):
                    if word.strip() == "<quant>":
                        input_text += " <quant> "
                    elif word == "," or word == "，":
                        input_text += word + " "
                    else:
                        input_text += word
            elif "MathQA" in file or "mawps" in file or "svamp" in file:
                input_text = ' '.join(mapped_text.split())
            else:
                raise NotImplementedError("The file type is not supported")
            res = tokenizer.encode_plus(" " + input_text, add_special_tokens=True, return_attention_mask=True)
            input_ids = res["input_ids"]
            attention_mask = res["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            var_starts = []
            var_ends = []
            quant_num = len(self.quant_list)
            # quants = ['<', 'q', '##uan', '##t', '>'] if not is_roberta_tokenizer else ['Ġ<', 'quant', '>']
            # obtain the start and end position of "<quant>" token
            for k, token in enumerate(tokens):
                if (token == self.quant_list[0]) and tokens[k:k + quant_num] == self.quant_list:
                    var_starts.append(k)
                    var_ends.append(k + quant_num - 1)

            assert len(input_ids) < 512 ## make sure no error in tokenization
            num_variable = len(var_starts)
            assert len(var_starts) == len(obj["num_list"])
            if len(obj["num_list"]) == 0:
                filter_type_count["no detected variable"] += 1
                obj['type_str'] = "no detected variable"
                continue
            var_mask = [1] * num_variable
            if len(obj["equation_layer"])  == 0:
                filter_type_count["empty equation in the data"]  += 1
                obj['type_str'] = "empty eqution"
                continue

            ##check duplication for (non-duplicated dataset, i.e., no same equation)
            if "nodup" in file:
                eq_set = set()
                for equation in obj["equation_layer"]:
                    eq_set.add(' '.join(equation))
                try:
                    assert len(eq_set) == len(obj["equation_layer"])
                except:
                    found_duplication_inst_num += 1

            labels = self.get_label_ids_incremental(obj["equation_layer"], add_replacement=True)

            if not labels:
                filter_type_count["cannot obtain the label sequence"] += 1
                obj['type_str'] = "illegal"
                continue
            # compute_value(labels, obj["num_list"])

            if len(labels) > self.data_max_height:
                filter_type_count[f"larger than the max height {self.data_max_height}"] += 1
                continue
            for left, right, _, _ in labels:
                assert left <= right

            if isinstance(labels, str):
                filter_type_count[f"index error for labels"] += 1
                obj['type_str'] = "illegal"
                continue
            try:
                res, _ = compute_value_for_incremental_equations(labels, obj["num_list"], self.constant_num, uni_labels=self.uni_labels, constant_values=self.constant_values)
            except:
                # print("answer calculate exception")
                filter_type_count[f"answer_calculate_exception"] += 1
                obj['type_str'] = "illegal"
                continue

            diff = res - float(obj["answer"])
            try:
                if float(obj["answer"]) > 1000000:
                    assert math.fabs(diff) < 200
                else:
                    assert math.fabs(diff) < 1
            except:
                # traceback.print_exc()
                obj['type_str'] = "illegal"
                if "test" in file or "valid" in file:
                    filter_type_count[f"answer not equal"] += 1
                    continue
                else:
                    if "MathQA" in file:
                        filter_type_count[f"answer not equal"] += 1
                        continue
                    else:
                        pass
            if filtered_steps is not None:
                if len(labels) not in filtered_steps:
                    filter_step_count += 1
                    continue

            label_height_mask = [1] * len(labels)
            num_step_count[len(labels)] += 1
            max_num_steps = max(max_num_steps, len(labels))
            ## check label all valid
            for label in labels:
                assert all([label[i] >= 0 for i in range(4)])
            equation_layer_num += len(obj["equation_layer"])
            equation_layer_num_count[len(obj["equation_layer"])] += 1
            sent_len_all += sent_len
            var_num_all += len(obj["num_list"])
            var_num_count[len(obj["num_list"])] += 1
            self._features.append(
                UniFeature(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids = [0] * len(input_ids),
                           variable_indexs_start=var_starts,
                           variable_indexs_end=var_ends,
                           num_variables=num_variable,
                           variable_index_mask=var_mask,
                           labels = labels,
                           label_height_mask=label_height_mask)
            )
            self.insts.append(obj)
        logger.info(f", total number instances: {len(self._features)} (before filter: {len(data)}), max num steps: {max_num_steps}")
        self.number_instances_remove = sum(filter_type_count.values())
        logger.info(f"filtered type counter: {filter_type_count}")
        logger.info(f"number of instances removed: {self.number_instances_remove}")
        assert self.number_instances_remove == len(data) - len(self._features)
        if found_duplication_inst_num:
            logger.warning(f"[WARNING] find duplication num: {found_duplication_inst_num} (not removed)")
        logger.debug(f"filter step count: {filtered_steps}")
        logger.info(num_step_count)
        avg_eq_num = equation_layer_num * 1.0/ len(self._features)
        logger.debug(f"average operation number: {avg_eq_num}, total: {equation_layer_num}, counter: {equation_layer_num_count}")
        avg_sent_len = sent_len_all * 1.0 / len(self._features)
        logger.debug(f"average sentence length: {avg_sent_len}, total: {sent_len_all}")
        logger.debug(f"variable number avg: {var_num_all * 1.0 / len(self._features)}, total: {var_num_all}, counter:{var_num_count}")

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> UniFeature:
        return self._features[idx]

    def get_label_ids_incremental(self, equation_layers: List, add_replacement: bool) -> Union[List[List[int]], None]:
        # in this data, only have one or zero bracket
        label_ids = []
        num_constant = len(self.constant2id) if self.constant2id is not None else 0
        for l_idx, layer in enumerate(equation_layers):
            left_var, right_var, op = layer
            if left_var == right_var and (not add_replacement):
                return None
            is_stop = 1 if l_idx == len(equation_layers) - 1 else 0
            if (not left_var.startswith("m_")):
                if self.constant2id is not None and left_var in self.constant2id:
                    left_var_idx = self.constant2id[left_var] + l_idx
                else:
                    try:
                        assert ord(left_var) >= ord('a') and ord(left_var) <= ord('z')
                    except:
                        print(f"[WARNING] find left_var ({left_var}) invalid, returning FALSE")
                        return None
                    left_var_idx = (ord(left_var) - ord('a') + num_constant + l_idx)
            else:
                m_idx = int(left_var[2:])
                # left_var_idx = -1
                left_var_idx = l_idx - m_idx
            if (not right_var.startswith("m_")):
                if self.constant2id is not None and right_var in self.constant2id:
                    right_var_idx = self.constant2id[right_var] + l_idx
                else:
                    try:
                        assert ord(right_var) >= ord('a') and ord(right_var) <= ord('z')
                    except:
                        print(f"[WARNING] find right var ({right_var}) invalid, returning FALSE")
                        return None
                    right_var_idx = (ord(right_var) - ord('a') + num_constant + l_idx)
            else:
                m_idx = int(right_var[2:])
                # left_var_idx = -1
                right_var_idx = l_idx - m_idx
            # try:
            assert right_var_idx >= 0
            # except:
            #     print("right var index error")
            #     return "right var index error"
            # try:
            assert left_var_idx >= 0
            # except:
            #     return "index error"

            if left_var.startswith("m_") or right_var.startswith("m_"):
                if left_var.startswith("m_") and (not right_var.startswith("m_")):
                    assert left_var_idx < right_var_idx
                    op_idx = self.uni_labels.index(op)
                    label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                elif not left_var.startswith("m_") and right_var.startswith("m_"):
                    assert left_var_idx > right_var_idx
                    op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else self.uni_labels.index(op[:-4])
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    ## both starts with m
                    if left_var_idx >= right_var_idx: ##left larger means left m_idx smaller
                        op = op[:-4] if left_var_idx == right_var_idx and op.endswith("_rev") else op
                        op_idx = self.uni_labels.index(op)
                        if left_var_idx > right_var_idx and (op not in ["+", "*"]):
                            op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else self.uni_labels.index(op[:-4])
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        #left < right
                        if (op in ["+", "*"]):
                            op_idx = self.uni_labels.index(op)
                            label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                        else:
                            # assert not op.endswith("_rev")
                            assert  "+" not in op and "*" not in op
                            op_idx = self.uni_labels.index(op)
                            label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                if left_var_idx <= right_var_idx:
                    if left_var_idx == right_var_idx and op.endswith("_rev"):
                        op = op[:-4]
                    op_idx = self.uni_labels.index(op)
                    label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                else:
                    # left > right
                    if (op in ["+", "*"]):
                        op_idx = self.uni_labels.index(op)
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        # assert not op.endswith("_rev")
                        op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else self.uni_labels.index(op[:-4])
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
        return label_ids

    def collate_function(self, batch: List[UniFeature]):

        max_wordpiece_length = max([len(feature.input_ids)  for feature in batch])
        max_num_variable = max([feature.num_variables  for feature in batch])
        max_height = max([len(feature.labels) for feature in batch])
        padding_value = [0,0,0,0]
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            attn_mask = feature.attention_mask + [0]* padding_length
            token_type_ids = feature.token_type_ids + [0]* padding_length
            padded_variable_idx_len = max_num_variable - feature.num_variables
            var_starts = feature.variable_indexs_start + [0] * padded_variable_idx_len
            var_ends = feature.variable_indexs_end + [0] * padded_variable_idx_len
            variable_index_mask = feature.variable_index_mask + [0] * padded_variable_idx_len

            padded_height = max_height - len(feature.labels)
            labels = feature.labels + [padding_value]* padded_height ## useless, because we have height mask
            label_height_mask = feature.label_height_mask + [0] * padded_height


            batch[i] = UniFeature(input_ids=np.asarray(input_ids),
                                attention_mask=np.asarray(attn_mask),
                                  token_type_ids=np.asarray(token_type_ids),
                                 variable_indexs_start=np.asarray(var_starts),
                                 variable_indexs_end=np.asarray(var_ends),
                                 num_variables=np.asarray(feature.num_variables),
                                 variable_index_mask=np.asarray(variable_index_mask),
                                 labels =np.asarray(labels),
                                  label_height_mask=np.asarray(label_height_mask))
        results = UniFeature(*(default_collate(samples) for samples in zip(*batch)))
        return results


def main_for_mawps():
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    constants = ['12.0', '1.0', '7.0', '60.0', '2.0', '5.0', '100.0', '8.0', '0.1', '0.5', '0.01', '25.0', '4.0', '3.0', '0.25']
    constant2id = {c: idx for idx, c in enumerate(constants)}
    constant_values = [float(c) for c in constants]
    pretrained_language_moel = 'roberta-base' ## bert-base-cased, roberta-base, bert-base-multilingual-cased, xlm-roberta-base
    tokenizer = class_name_2_tokenizer[pretrained_language_moel].from_pretrained(pretrained_language_moel)
    test_dataset = UniversalDataset(file="../../data/mawps-single/mawps_test_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                                    pretrained_model_name=pretrained_language_moel)
    train_dataset = UniversalDataset(file="../../data/mawps-single/mawps_train_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                                     pretrained_model_name=pretrained_language_moel)
    validation_dataset = UniversalDataset(file="../../data/mawps-single/mawps_valid_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                                          pretrained_model_name=pretrained_language_moel)

def main_for_svamp():
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    constants = ['1.0', '0.1', '3.0', '5.0', '0.5', '12.0', '4.0', '60.0', '25.0', '0.01', '0.05', '2.0', '10.0', '0.25', '8.0', '7.0', '100.0']
    # constants = ['0.01', '12.0', '1.0', '100.0', '0.1', '0.5', '3.0', '4.0', '7.0']
    constant2id = {c: idx for idx, c in enumerate(constants)}
    constant_values = [float(c) for c in constants]
    pretrained_language_moel = 'roberta-base' ## bert-base-cased, roberta-base, bert-base-multilingual-cased, xlm-roberta-base
    tokenizer = class_name_2_tokenizer[pretrained_language_moel].from_pretrained(pretrained_language_moel)
    UniversalDataset(file="../../data/mawps_asdiv-a_svamp/testset_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/mawps_asdiv-a_svamp/trainset_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_moel)

def main_for_math23k():
    pretrained_language_model = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(pretrained_language_model)
    constant2id = {"1": 0, "PI": 1}
    constant_values = [1.0, 3.14]
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    # uni_labels += ["^", "^_rev"]
    data_max_height = 15
    UniversalDataset(file="../../data/math23k/test23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_model,
                     data_max_height = data_max_height)
    UniversalDataset(file="../../data/math23k/train23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_model,
                     data_max_height=data_max_height, filtered_steps=None)
    UniversalDataset(file="../../data/math23k/valid23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_model,
                     data_max_height=data_max_height)

def main_for_mathqa():
    # constants = ['3.6', '2.0', '52.0', '3600.0', '12.0', '1.0', '60.0', '0.3937',
    #              '3.0', '4.0', '3.1416', '1.6', '0.33', '5.0', '360.0', '26.0', '1000.0',
    #              '100.0', '0.5', '180.0', '0.25', '10.0', '6.0', '0.01745', '0.2778']
    constants = ['100.0', '1.0', '2.0', '3.0', '4.0', '10.0', '1000.0', '60.0', '0.5', '3600.0', '12.0', '0.2778', '3.1416', '3.6',
     '0.25', '5.0', '6.0', '360.0', '52.0', '180.0']
    ###[('100.0', 8184), ('1.0', 5872), ('2.0', 5585), ('3.0', 1803), ('4.0', 1453), ('10.0', 773), ('1000.0', 732),
    # ('60.0', 621), ('0.5', 595), ('3600.0', 412), ('12.0', 217), ('0.2778', 154), ('3.1416', 134), ('3.6', 104), ('0.25', 64),
    # ('5.0', 35), ('6.0', 31), ('360.0', 9), ('52.0', 6), ('180.0', 5), ('26.0', 3), ('1.6', 3), ('0.33', 2), ('0.3937', 1)]

    constant2id = {c: idx for idx, c in enumerate(constants)}
    constant_values = [float(c) for c in constants]
    uni_labels = [
            '+', '-', '-_rev', '*', '/', '/_rev'
        ]
    uni_labels = uni_labels + ['^', '^_rev']
    pretrained_language_moel = 'roberta-base'
    data_max_height=15
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_language_moel)
    # UniversalDataset(file="../../data/MathQA/debug.json", tokenizer=tokenizer, uni_labels=uni_labels,
    #                  constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
    #                  use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/MathQA/mathqa_test_nodup_our_filtered.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/MathQA/mathqa_dev_nodup_our_filtered.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/MathQA/mathqa_train_nodup_our_filtered.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values,
                     pretrained_model_name=pretrained_language_moel)

if __name__ == '__main__':
    logger.addHandler(logging.StreamHandler())
    from transformers import BertTokenizer, RobertaTokenizerFast, XLMRobertaTokenizerFast
    add_replacement = True
    use_incremental_labeling = True
    class_name_2_tokenizer = {
        "bert-base-cased": BertTokenizerFast,
        "roberta-base": RobertaTokenizerFast,
        "bert-base-multilingual-cased": BertTokenizerFast,
        "xlm-roberta-base": XLMRobertaTokenizerFast,
        'hfl/chinese-bert-wwm-ext': BertTokenizerFast,
        'hfl/chinese-roberta-wwm-ext': BertTokenizerFast,
    }
    # main_for_svamp()
    # main_for_mawps()
    # main_for_mathqa()
    main_for_math23k()



