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
from src.eval.utils import compute_value, compute_value_for_incremental_equations, compute_value_for_parallel_equations
import math
from typing import Dict, List
from collections import Counter

"""
not finished yet
"""

class_name_2_quant_list = {
    "bert-base-cased": ['<', 'q', '##uant', '>'],
    "roberta-base": ['Ġ<', 'quant', '>'],
    "bert-base-multilingual-cased": ['<', 'quant', '>'],
    "xlm-roberta-base": ['▁<', 'quant', '>'],
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
                 number: int = -1, add_replacement: bool = False,
                 filtered_steps: List = None,
                 constant2id: Dict[str, int] = None,
                 constant_values: List[float] = None,
                 use_incremental_labeling: bool = False,
                 add_new_token: bool = False,
                 data_max_height: int = 100,
                 test_strings: List[str] = None) -> None:
        self.tokenizer = tokenizer
        self.constant2id = constant2id
        self.constant_values = constant_values
        self.constant_num = len(self.constant2id) if self.constant2id else 0
        self.use_incremental_labeling = use_incremental_labeling
        self.add_new_token = add_new_token
        self.add_replacement = add_replacement
        self.data_max_height = data_max_height
        self.uni_labels = uni_labels
        self.quant_list = class_name_2_quant_list[pretrained_model_name]
        if file is not None:
            self.read_math23k_file(file, tokenizer, number, add_replacement, filtered_steps)
        else:
            self._features = []
            for sent in test_strings:
                for k in range(ord('a'), ord('a') + 26):
                    sent = sent.replace(f"temp_{chr(k)}", " <quant> ")
                res = tokenizer.encode_plus(" " + sent, add_special_tokens=True, return_attention_mask=True)

                input_ids = res["input_ids"]
                attention_mask = res["attention_mask"]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                var_starts = []
                var_ends = []
                quant_num = len(self.quant_list)
                for k, token in enumerate(tokens):
                    if (token == self.quant_list[0]) and tokens[k:k + quant_num] == self.quant_list:
                        var_starts.append(k)
                        var_ends.append(k + quant_num - 1)
                num_variable = len(var_starts)
                var_mask = [1] * len(var_starts)
                labels = [[-100, -100, -100, -100]]
                label_height_mask = [0]
                self._features.append(
                    UniFeature(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=[0] * len(input_ids),
                               variable_indexs_start=var_starts,
                               variable_indexs_end=var_ends,
                               num_variables=num_variable,
                               variable_index_mask=var_mask,
                               labels=labels,
                               label_height_mask=label_height_mask)
                )


    def read_math23k_file(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1, add_replacement: bool = False,
                 filtered_steps: List = None) -> None:
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        num_cant_find_labels = 0
        max_num_steps = 0
        self.insts = []
        num_index_error = 0
        number_instances_filtered = 0
        number_instances_more_than_max_height_filtered= 0
        num_step_count = Counter()
        num_empty_equation = 0
        max_intermediate_num_for_parallel = 0
        num_mawps_constant_removed = 0
        not_equal_num = 0
        answer_calculate_exception = 0
        num_var_is_zero = 0
        equation_layer_num = 0
        equation_layer_num_count = Counter()
        var_num_all =0
        var_num_count = Counter()
        sent_len_all = 0
        filter_type_count = Counter()
        found_duplication_inst_num = 0
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            if obj['type_str'] != "legal" and obj['type_str'] != "variable more than 7":
                filter_type_count[obj["type_str"]] += 1
                number_instances_filtered += 1
                continue
            mapped_text = obj["text"]
            sent_len = len(mapped_text.split())
            for k in range(ord('a'), ord('a') + 26):
                mapped_text = mapped_text.replace(f"temp_{chr(k)}", " <quant> ")
            if "math23k" in file or "large_math" in file:
                mapped_text = mapped_text.split()
                input_text = ""
                for idx, word in enumerate(mapped_text):
                    if word.strip() == "<quant>":
                        input_text += " <quant> "
                    elif word == "," or word == "，":
                        input_text += word + " "
                    else:
                        input_text += word
            elif "MathQA" in file or "mawps" in file:
                input_text = ' '.join(mapped_text.split())
            else:
                raise NotImplementedError
            res = tokenizer.encode_plus(" " + input_text, add_special_tokens=True, return_attention_mask=True)
            input_ids = res["input_ids"]
            attention_mask = res["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            var_starts = []
            var_ends = []
            quant_num = len(self.quant_list)
            # quants = ['<', 'q', '##uan', '##t', '>'] if not is_roberta_tokenizer else ['Ġ<', 'quant', '>']
            if self.add_new_token:
                new_tokens = []
                k = 0
                while k < len(tokens):
                    curr_tok = tokens[k]
                    if (curr_tok == self.quant_list[0]) and tokens[k:k+quant_num] == self.quant_list:
                        new_tokens.append('<NUM>')
                        var_starts.append(len(new_tokens))
                        var_ends.append(len(new_tokens))
                        k = k+quant_num - 1
                    else:
                        new_tokens.append(curr_tok)
                    k+=1
                input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
                attention_mask = [1] * len(input_ids)
            else:
                for k, token in enumerate(tokens):
                    if (token == self.quant_list[0]) and tokens[k:k+quant_num] == self.quant_list:
                        var_starts.append(k)
                        var_ends.append(k+quant_num-1)
            assert len(input_ids) < 512
            num_variable = len(var_starts)
            assert len(var_starts) == len(obj["num_list"])
            if len(obj["num_list"]) == 0:
                num_var_is_zero += 1
                obj['type_str'] = "no detected variable"
                continue
            var_mask = [1] * num_variable
            if len(obj["equation_layer"])  == 0:
                num_empty_equation += 1
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

            if self.use_incremental_labeling:
                labels = self.get_label_ids_incremental(obj["equation_layer"], add_replacement=add_replacement)
            else:
                labels = self.get_label_ids_updated(obj["equation_layer"], add_replacement=add_replacement)

            if not labels:
                num_cant_find_labels += 1
                obj['type_str'] = "illegal"
                continue
            # compute_value(labels, obj["num_list"])

            if len(labels) > self.data_max_height:
                number_instances_more_than_max_height_filtered += 1
                continue
            if "parallel" in file:
                for equations in labels:
                    for left, right, _, _ in equations:
                        assert left <= right
            else:
                for left, right, _, _ in labels:
                    assert left <= right


            if isinstance(labels, str):
                num_index_error += 1
                obj['type_str'] = "illegal"
                continue
            try:
                if self.use_incremental_labeling:
                    res, _ = compute_value_for_incremental_equations(labels, obj["num_list"], self.constant_num, uni_labels=self.uni_labels, constant_values=self.constant_values)
                else:
                    res = compute_value(labels, obj["num_list"], self.constant_num, uni_labels=self.uni_labels, constant_values=self.constant_values)
            except:
                # print("answer calculate exception")
                answer_calculate_exception += 1
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
                # print("not equal", flush=True)
                not_equal_num += 1
                obj['type_str'] = "illegal"
                if "test" in file or "valid" in file:
                    continue
                else:
                    if "MathQA" in file:
                        continue
                    else:
                        pass
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
        print(f"number of instances that cannot find labels in m0: {num_cant_find_labels}, empty_equation: {num_empty_equation}, total number instances: {len(self._features)},"
              f"max num steps: {max_num_steps}, number_instances_filtered: {number_instances_filtered}, num_index_error: {num_index_error}")
        print(f"filtered type counter: {filter_type_count}")
        print(f"num mawps constant removed: {num_mawps_constant_removed}")
        print(f"totla number of answer not equal (skipping): {not_equal_num}, answer calculate exception: {answer_calculate_exception}")
        print(f"number of instances that have more than max height filtered: {number_instances_more_than_max_height_filtered}")
        print(f"number of instances with no detected variables: {num_var_is_zero}")
        print(f"[WARNING] find duplication num: {found_duplication_inst_num} (not removed)")
        print(num_step_count)
        avg_eq_num = equation_layer_num * 1.0/ len(self._features)
        print(f"average operation number: {avg_eq_num}, total: {equation_layer_num}, counter: {equation_layer_num_count}")
        avg_sent_len = sent_len_all * 1.0 / len(self._features)
        print(f"average sentence length: {avg_sent_len}, total: {sent_len_all}")
        print(f"variable number avg: {var_num_all * 1.0 / len(self._features)}, total: {var_num_all}, counter:{var_num_count}")
        ### out_file = "../../data/large_math/mwp_processed_filtered.json"
        ### write_data(file=out_file, data= data)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> UniFeature:
        return self._features[idx]


    def get_label_ids(self, equation_layers: List, add_replacement: bool) -> Union[List[List[int]], None]:
        # in this data, only have one or zero bracket
        label_ids = []
        for l_idx, layer in enumerate(equation_layers):
            left_var, right_var, op = layer
            if left_var == right_var and (not add_replacement):
                return None
            is_stop = 1 if l_idx == len(equation_layers) - 1 else 0

            left_var_idx = (ord(left_var) - ord('a')) if left_var != "#" else -1
            right_var_idx = (ord(right_var) - ord('a'))
            assert right_var_idx >= 0
            try:
                assert left_var_idx >=0 or left_var_idx == -1
            except:
                return "index error"
            if left_var_idx < right_var_idx:
                op_idx = self.uni_labels.index(op)
                label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                # left > right
                if op in ["+", "*"]:
                    op_idx = self.uni_labels.index(op)
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    assert not op.endswith("_rev")
                    op_idx = self.uni_labels.index(op + "_rev")
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
        return label_ids


    def get_label_ids_updated(self, equation_layers: List, add_replacement: bool) -> Union[List[List[int]], None]:
        # in this data, only have one or zero bracket
        label_ids = []
        num_constant = len(self.constant2id) if self.constant2id is not None else 0
        for l_idx, layer in enumerate(equation_layers):
            left_var, right_var, op = layer
            if left_var == right_var and (not add_replacement):
                return None
            is_stop = 1 if l_idx == len(equation_layers) - 1 else 0

            if left_var != "#" and (not left_var.startswith("m_")):
                if self.constant2id is not None and left_var in self.constant2id:
                    left_var_idx = self.constant2id[left_var]
                else:
                    # try:
                    assert ord(left_var) >= ord('a') and ord(left_var) <= ord('z')
                    # except:
                    #     print("seohting")
                    left_var_idx = (ord(left_var) - ord('a') + num_constant)
            else:
                left_var_idx = -1
            right_var_idx = (ord(right_var) - ord('a') + num_constant) if self.constant2id is None or (right_var not in self.constant2id) else self.constant2id[right_var]
            # try:
            assert right_var_idx >= 0
            # except:
            #     print("right var index error")
            #     return "right var index error"
            # try:
            assert left_var_idx >= -1
            # except:
            #     return "index error"
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
                    op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else  self.uni_labels.index(op[:-4])
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
        return label_ids


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
        padding_value = [-1, 0, 0, 0] if not self.use_incremental_labeling else [0,0,0,0]
        if self.use_incremental_labeling and not self.add_replacement:
            padding_value = [0, 1, 0, 0]
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
    UniversalDataset(file="../../data/mawps-single/mawps_test_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/mawps-single/mawps_train_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/mawps-single/mawps_valid_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)

def main_for_math23k():
    pretrained_language_model = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(pretrained_language_model)
    constant2id = {"1": 0, "PI": 1}
    constant_values = [1.0, 3.14]
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    data_max_height = 15
    UniversalDataset(file="../../data/math23k/test23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_model,
                     data_max_height = data_max_height)
    UniversalDataset(file="../../data/math23k/train23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_model,
                     data_max_height=data_max_height)
    UniversalDataset(file="../../data/math23k/valid23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_model,
                     data_max_height=data_max_height)

def main_for_ours():
    pretrained_language_moel = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(pretrained_language_moel)
    constants = ['5.0', '10.0', '2.0', '8.0', '30.0', '1.0', '6.0', '7.0', '12.0', '4.0', '31.0', '3.14', '3.0']
    constant2id = {c: idx for idx, c in enumerate(constants)}
    constant_values = [float(c) for c in constants]
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    data_max_height = 10
    # UniversalDataset(file="../../data/large_math/mwp_processed_extracted.json", tokenizer=tokenizer, uni_labels=uni_labels,
    #                  constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
    #                  use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/large_math/large_math_train_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel,
                     data_max_height=data_max_height)
    UniversalDataset(file="../../data/large_math/large_math_valid_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel,
                     data_max_height=data_max_height)
    UniversalDataset(file="../../data/large_math/large_math_test_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel,
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
    UniversalDataset(file="../../data/MathQA/mathqa_test_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement, data_max_height=data_max_height,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/MathQA/mathqa_dev_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement, data_max_height=data_max_height,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)
    UniversalDataset(file="../../data/MathQA/mathqa_train_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement, data_max_height=data_max_height,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False, pretrained_model_name=pretrained_language_moel)

if __name__ == '__main__':
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
    # main_for_mawps()
    main_for_ours()
    # main_for_mathqa()
    # main_for_math23k()



