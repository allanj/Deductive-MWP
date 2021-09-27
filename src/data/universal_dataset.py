import torch
from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizerFast, MBartTokenizerFast
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
uni_labels = [
    '+','-', '-_rev', '*', '/', '/_rev'
]

UniFeature = collections.namedtuple('UniFeature', 'input_ids attention_mask token_type_ids variable_indexs_start variable_indexs_end num_variables variable_index_mask labels label_height_mask')
UniFeature.__new__.__defaults__ = (None,) * 7

class UniversalDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1, add_replacement: bool = False,
                 filtered_steps: List = None,
                 constant2id: Dict[str, int] = None,
                 constant_values: List[float] = None,
                 use_incremental_labeling: bool = False,
                 add_new_token: bool = False) -> None:
        self.tokenizer = tokenizer
        self.constant2id = constant2id
        self.constant_values = constant_values
        self.constant_num = len(self.constant2id) if self.constant2id else 0
        self.use_incremental_labeling = use_incremental_labeling
        self.add_new_token = add_new_token
        if "complex" in file:
            self.read_complex_file(file, tokenizer, number, add_replacement, filtered_steps)
        else:
            self.read_math23k_file(file, tokenizer, number, add_replacement, filtered_steps)

    def read_complex_file(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1, add_replacement: bool = False,
                 filtered_steps: List = None) -> None:
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        num_has_same_var_m0 = 0
        max_num_steps = 0
        self.insts = []
        filtered_steps = [int(f) for f in filtered_steps] if filtered_steps else None
        numbert_instances_filtered = 0
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            # if not (obj['legal'] and obj['num_steps'] <= 2):
            #     continue
            if not obj['legal']:
                continue
            if filtered_steps and obj['num_steps'] in filtered_steps:
                ## in experiments, we can choose to filter some questions with specific steps
                numbert_instances_filtered += 1
                continue
            ## equation preprocessing
            # mapped_equation = obj["mapped_equation"]
            # for k in range(ord('a'), ord('a') + 26):
            #     mapped_equation = mapped_equation.replace(f"( temp_{chr(k)} )", f"temp_{chr(k)}")
            # pattern = r"\( ?\( ?temp_\w [\+\-\*\/] temp_\w ?\) ?\)"
            # if len(re.findall(pattern, mapped_equation)) > 0:
            #     mapped_equation = mapped_equation.replace("( ( ", "( ").replace(") ) ", ") ")
            ### end equation preprocess
            mapped_text = obj["mapped_text"]
            for k in range(ord('a'), ord('a') + 26):
                mapped_text = mapped_text.replace(f"@ {chr(k)}", " <quant> ")
            mapped_text = mapped_text.split()
            input_text = ""
            for idx, word in enumerate(mapped_text):
                if word.strip() == "<quant>":
                    input_text += " <quant> "
                elif word == "," or word == "，":
                    input_text += word + " "
                else:
                    input_text += word
            res = tokenizer.encode_plus(input_text, add_special_tokens=True, return_attention_mask=True)
            input_ids = res["input_ids"]
            attention_mask = res["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            var_starts = []
            var_ends = []
            for k, token in enumerate(tokens):
                if token == "<" and tokens[k:k+5] == ['<', 'q', '##uan', '##t', '>']:
                    var_starts.append(k)
                    var_ends.append(k+4)
            num_variable = len(var_starts)
            var_mask = [1] * num_variable
            labels = self.get_label_ids(obj["equation_layer"], add_replacement=add_replacement)
            if not labels:
                num_has_same_var_m0 += 1
                continue
            label_height_mask = [1] * len(labels)
            max_num_steps = max(max_num_steps, len(labels))
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
        print(f"number of instances that have same variable in m0: {num_has_same_var_m0}, total number instances: {len(self._features)},"
              f"max num steps: {max_num_steps}, numbert_instances_filtered: {numbert_instances_filtered}")

    def read_math23k_file(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1, add_replacement: bool = False,
                 filtered_steps: List = None) -> None:
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        num_has_same_var_m0 = 0
        max_num_steps = 0
        self.insts = []
        num_index_error = 0
        numbert_instances_filtered = 0
        num_step_count = Counter()
        num_empty_equation = 0
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            if obj['type_str'] != "legal":
                numbert_instances_filtered += 1
                continue
            mapped_text = obj["text"]
            for k in range(ord('a'), ord('a') + 26):
                mapped_text = mapped_text.replace(f"temp_{chr(k)}", " <quant> ")
            mapped_text = mapped_text.split()
            input_text = ""
            for idx, word in enumerate(mapped_text):
                if word.strip() == "<quant>":
                    input_text += " <quant> "
                elif word == "," or word == "，":
                    input_text += word + " "
                else:
                    input_text += word
            res = tokenizer.encode_plus(input_text, add_special_tokens=True, return_attention_mask=True)
            input_ids = res["input_ids"]
            attention_mask = res["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            var_starts = []
            var_ends = []
            if self.add_new_token:
                new_tokens = []
                k = 0
                while k < len(tokens):
                    curr_tok = tokens[k]
                    if curr_tok == "<" and tokens[k:k+5] == ['<', 'q', '##uan', '##t', '>']:
                        new_tokens.append('<NUM>')
                        var_starts.append(len(new_tokens))
                        var_ends.append(len(new_tokens))
                        k = k+4
                    else:
                        new_tokens.append(curr_tok)
                    k+=1
                input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
                attention_mask = [1] * len(input_ids)
            else:
                for k, token in enumerate(tokens):
                    if token == "<" and tokens[k:k+5] == ['<', 'q', '##uan', '##t', '>']:
                        var_starts.append(k)
                        var_ends.append(k+4)
            num_variable = len(var_starts)
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
                    print("[WARNING] [Probably ERROR] find duplication")

            if self.use_incremental_labeling:
                if "parallel" in file:
                    labels = self.get_label_ids_parallel(obj["sorted_equation_layer"], add_replacement=add_replacement)
                else:
                    labels = self.get_label_ids_incremental(obj["equation_layer"], add_replacement=add_replacement)
            else:
                labels = self.get_label_ids_updated(obj["equation_layer"], add_replacement=add_replacement)

            if not labels:
                num_has_same_var_m0 += 1
                obj['type_str'] = "illegal"
                continue
            # compute_value(labels, obj["num_list"])

            if len(labels) > 10 and "test" in file:
                numbert_instances_filtered += 1
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
            if self.use_incremental_labeling:
                if "parallel" in file:
                    res, _ = compute_value_for_parallel_equations(labels, obj["num_list"], self.constant_num, constant_values=self.constant_values)
                else:
                    res, _ = compute_value_for_incremental_equations(labels, obj["num_list"], self.constant_num, constant_values=self.constant_values)
            else:
                res = compute_value(labels, obj["num_list"], self.constant_num, constant_values=self.constant_values)


            try:
                if obj["answer"] > 1000000:
                    assert math.fabs(res - obj["answer"]) < 200
                else:
                    assert math.fabs(res - obj["answer"]) < 1e-4
            except:
                print("not equal", flush=True)
            label_height_mask = [1] * len(labels)
            num_step_count[len(labels)] += 1

            max_num_steps = max(max_num_steps, len(labels))
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
        print(f"number of instances that have same variable in m0: {num_has_same_var_m0}, empty_equation: {num_empty_equation}, total number instances: {len(self._features)},"
              f"max num steps: {max_num_steps}, numbert_instances_filtered: {numbert_instances_filtered}, num_index_error: {num_index_error}")
        print(num_step_count)
        # out_file = file.split(".json")[0] + "_baseline.json"
        # write_data(file=out_file, data= data)

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
                op_idx = uni_labels.index(op)
                label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                # left > right
                if op in ["+", "*"]:
                    op_idx = uni_labels.index(op)
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    assert not op.endswith("_rev")
                    op_idx = uni_labels.index(op + "_rev")
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
                op_idx = uni_labels.index(op)
                label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                # left > right
                if (op in ["+", "*"]):
                    op_idx = uni_labels.index(op)
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    # assert not op.endswith("_rev")
                    op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else  uni_labels.index(op[:-4])
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
                    # try:
                    assert ord(left_var) >= ord('a') and ord(left_var) <= ord('z')
                    # except:
                    #     print("seohting")
                    left_var_idx = (ord(left_var) - ord('a') + num_constant + l_idx)
            else:
                m_idx = int(left_var[2:])
                # left_var_idx = -1
                left_var_idx = l_idx - m_idx
            if (not right_var.startswith("m_")):
                if self.constant2id is not None and right_var in self.constant2id:
                    right_var_idx = self.constant2id[right_var] + l_idx
                else:
                    assert ord(right_var) >= ord('a') and ord(right_var) <= ord('z')
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
                    op_idx = uni_labels.index(op)
                    label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                elif not left_var.startswith("m_") and right_var.startswith("m_"):
                    assert left_var_idx > right_var_idx
                    op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    ## both starts with m
                    if left_var_idx >= right_var_idx: ##left larger means left m_idx smaller
                        op = op[:-4] if left_var_idx == right_var_idx and op.endswith("_rev") else op
                        op_idx = uni_labels.index(op)
                        if left_var_idx > right_var_idx and (op not in ["+", "*"]):
                            op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        #left < right
                        if (op in ["+", "*"]):
                            op_idx = uni_labels.index(op)
                            label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                        else:
                            # assert not op.endswith("_rev")
                            assert  "+" not in op and "*" not in op
                            op_idx = uni_labels.index(op)
                            label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                if left_var_idx <= right_var_idx:
                    if left_var_idx == right_var_idx and op.endswith("_rev"):
                        op = op[:-4]
                    op_idx = uni_labels.index(op)
                    label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                else:
                    # left > right
                    if (op in ["+", "*"]):
                        op_idx = uni_labels.index(op)
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        # assert not op.endswith("_rev")
                        op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
        return label_ids


    def get_label_ids_parallel(self, parallel_equation_layers: List[List], add_replacement: bool) -> Union[List[List[List[int]]], None]:
        # in this data, only have one or zero bracket
        all_label_ids = []
        num_constant = len(self.constant2id) if self.constant2id is not None else 0
        accumulate_eqs = [0]
        for p_idx, equation_layers in enumerate(parallel_equation_layers):
            current_label_ids = []
            for l_idx, layer in enumerate(equation_layers):
                left_var, right_var, op = layer
                if left_var == right_var and (not add_replacement):
                    return None
                is_stop = 1 if l_idx == len(equation_layers) - 1 and p_idx == len(parallel_equation_layers) - 1 else 0
                if (not left_var.startswith("m_")):
                    if self.constant2id is not None and left_var in self.constant2id:
                        left_var_idx = self.constant2id[left_var] + accumulate_eqs[p_idx]
                    else:
                        # try:
                        assert ord(left_var) >= ord('a') and ord(left_var) <= ord('z')
                        # except:
                        #     print("seohting")
                        left_var_idx = (ord(left_var) - ord('a') + num_constant + accumulate_eqs[p_idx])
                else:
                    _, m_p_idx, m_v_idx = left_var.split("_")
                    m_p_idx, m_v_idx = int(m_p_idx), int(m_v_idx)
                    # left_var_idx = -1
                    left_var_idx = accumulate_eqs[p_idx] - accumulate_eqs[m_p_idx+1]  + m_v_idx
                if (not right_var.startswith("m_")):
                    if self.constant2id is not None and right_var in self.constant2id:
                        right_var_idx = self.constant2id[right_var] + accumulate_eqs[p_idx]
                    else:
                        assert ord(right_var) >= ord('a') and ord(right_var) <= ord('z')
                        right_var_idx = (ord(right_var) - ord('a') + num_constant + accumulate_eqs[p_idx])
                else:
                    _, m_p_idx, m_v_idx = right_var.split("_")
                    m_p_idx, m_v_idx = int(m_p_idx), int(m_v_idx)
                    right_var_idx = accumulate_eqs[p_idx] - accumulate_eqs[m_p_idx+1] + m_v_idx
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
                        op_idx = uni_labels.index(op)
                        current_label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                    elif not left_var.startswith("m_") and right_var.startswith("m_"):
                        assert left_var_idx > right_var_idx
                        op_idx = uni_labels.index(op)
                        if op not in ["+", "*"]:
                            op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                        current_label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        ## both starts with m
                        if left_var_idx >= right_var_idx: ##left larger means left m_idx smaller
                            op = op[:-4] if left_var_idx == right_var_idx and op.endswith("_rev") else op
                            op_idx = uni_labels.index(op)
                            if left_var_idx > right_var_idx and (op not in ["+", "*"]):
                                op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                            current_label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                        else:
                            #left < right
                            if (op in ["+", "*"]):
                                op_idx = uni_labels.index(op)
                                current_label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                            else:
                                # assert not op.endswith("_rev")
                                assert  "+" not in op and "*" not in op
                                op_idx = uni_labels.index(op)
                                current_label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                else:
                    if left_var_idx <= right_var_idx:
                        if left_var_idx == right_var_idx and op.endswith("_rev"):
                            op = op[:-4]
                        op_idx = uni_labels.index(op)
                        current_label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                    else:
                        # left > right
                        if (op in ["+", "*"]):
                            op_idx = uni_labels.index(op)
                            current_label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                        else:
                            # assert not op.endswith("_rev")
                            op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                            current_label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
            all_label_ids.append(current_label_ids)
            accumulate_eqs.append(accumulate_eqs[len(accumulate_eqs)-1] + len(equation_layers))
        return all_label_ids

    def collate_function(self, batch: List[UniFeature]):

        max_wordpiece_length = max([len(feature.input_ids)  for feature in batch])
        max_num_variable = max([feature.num_variables  for feature in batch])
        max_height = max([len(feature.labels) for feature in batch])
        padding_value = [-1, 0, 0, 0] if not self.use_incremental_labeling else [0,0,0,0]
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

    def update_gold_idx(self, gold_l, padded_indexs, current_prev_intermediate_num, curr_accumate_prev_var, curr_prev_pad, curr_max_accumulate_prev):
        if gold_l >= current_prev_intermediate_num and gold_l < curr_accumate_prev_var:
            if len(padded_indexs) > 0:
                curr_min_m_idx = current_prev_intermediate_num - 1
                for k in range(current_prev_intermediate_num, curr_max_accumulate_prev):
                    if k not in padded_indexs:
                        curr_min_m_idx += 1
                        if curr_min_m_idx == gold_l:
                            gold_l = k
                            break
                return gold_l
        elif gold_l >= curr_accumate_prev_var:
            gold_l += curr_prev_pad
        return gold_l

    def collate_parallel(self, batch: List[UniFeature]):

        max_wordpiece_length = max([len(feature.input_ids)  for feature in batch])
        max_num_variable = max([feature.num_variables  for feature in batch])
        max_height = max([len(feature.labels) for feature in batch])
        max_prev_num_intermediate = [0]
        accumulate_max_prev_num_intermediate = [0]
        for h in range(max_height):
            curr_num_intermediates = [0] # for later max(), to appear no exception, because max([]) not valid.
            for feature in batch:
                curr_num_intermediates.append(len(feature.labels[h]) if len(feature.labels) > h else 0)
            max_prev_num_intermediate.append(max(curr_num_intermediates))
            accumulate_max_prev_num_intermediate.append(max_prev_num_intermediate[-1] + accumulate_max_prev_num_intermediate[-1])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            attn_mask = feature.attention_mask + [0]* padding_length
            token_type_ids = feature.token_type_ids + [0]* padding_length
            padded_variable_idx_len = max_num_variable - feature.num_variables
            var_starts = feature.variable_indexs_start + [0] * padded_variable_idx_len
            var_ends = feature.variable_indexs_end + [0] * padded_variable_idx_len
            variable_index_mask = feature.variable_index_mask + [0] * padded_variable_idx_len

            current_labels = []
            padded_labels_at_comb =  [[-100, -100] for _ in enumerate(uni_labels)]
            curr_accumate_prev_var = 0
            padded_indexs = []
            for h_idx in range(max_height):

                current_labels_at_h = []
                current_max_prev_num_intermediate = max_prev_num_intermediate[h_idx]
                num_var_for_comb = accumulate_max_prev_num_intermediate[h_idx] + self.constant_num + max_num_variable
                num_var_range = torch.arange(0, num_var_for_comb)
                combination = torch.combinations(num_var_range, r=2, with_replacement=self.use_incremental_labeling) ##number_of_combinations x 2
                num_combinations, _ = combination.size()
                combination = combination.numpy()
                curr_accumate_prev_var += len(feature.labels[h_idx - 1]) if h_idx > 0 and h_idx < len(feature.labels) else 0
                curr_prev_pad = accumulate_max_prev_num_intermediate[h_idx] - curr_accumate_prev_var

                for pid in range(len(padded_indexs)):
                    padded_indexs[pid] += current_max_prev_num_intermediate

                if h_idx >= len(feature.labels):
                    current_labels_at_h = np.full((num_combinations, len(uni_labels), 2), -100).tolist()
                else:
                    current_prev_intermediate_num = len(feature.labels[h_idx - 1]) if h_idx > 0 else 0
                    current_gold_labels = sorted(feature.labels[h_idx])  ## list of equations
                    c_idx = 0
                    for comb_idx, comb in enumerate(combination):
                        left, right = comb
                        current_labels_at_comb = None
                        if left >= current_prev_intermediate_num and left < current_max_prev_num_intermediate:
                            # padded intermediate var
                            current_labels_at_comb = padded_labels_at_comb
                            if left not in padded_indexs:
                                padded_indexs.append(left)
                        elif left >= current_max_prev_num_intermediate and left < accumulate_max_prev_num_intermediate[h_idx] and left in padded_indexs:
                            current_labels_at_comb = padded_labels_at_comb
                        elif left >= accumulate_max_prev_num_intermediate[h_idx] + self.constant_num + feature.num_variables:
                            ## pad maximum
                            current_labels_at_comb = padded_labels_at_comb

                        if right >= current_prev_intermediate_num and right < current_max_prev_num_intermediate:
                            # padded intermediate var
                            current_labels_at_comb = padded_labels_at_comb
                            if right not in padded_indexs:
                                padded_indexs.append(right)
                        elif right >= current_max_prev_num_intermediate and right < accumulate_max_prev_num_intermediate[h_idx] and right in padded_indexs:
                            current_labels_at_comb = padded_labels_at_comb
                        elif right >= accumulate_max_prev_num_intermediate[h_idx] + self.constant_num + feature.num_variables:
                            ## pad maximum
                            current_labels_at_comb = padded_labels_at_comb

                        if current_labels_at_comb is None: ## that means left, right is valid
                            if c_idx < len(current_gold_labels):
                                current_labels_at_comb = []
                                gold_l, gold_r, gold_label, gold_stop = current_gold_labels[c_idx]
                                updated_gold_l = self.update_gold_idx(gold_l, padded_indexs, current_prev_intermediate_num, curr_accumate_prev_var, curr_prev_pad, accumulate_max_prev_num_intermediate[h_idx])
                                updated_gold_r = self.update_gold_idx(gold_r, padded_indexs, current_prev_intermediate_num, curr_accumate_prev_var, curr_prev_pad, accumulate_max_prev_num_intermediate[h_idx])
                                should_exit = False
                                for label_idx, _ in enumerate(uni_labels):
                                    current_labels_at_labels = [0] * 2 ## stop label
                                    if should_exit:
                                        current_labels_at_comb.append(current_labels_at_labels)
                                        continue
                                    if [updated_gold_l, updated_gold_r, gold_label, gold_stop] == [left, right, label_idx, 0]:
                                        current_labels_at_labels[0] = 1
                                        c_idx += 1
                                        if c_idx < len(current_gold_labels):
                                            next_gold_l, next_gold_r, gold_label, gold_stop = current_gold_labels[c_idx]
                                            if not (next_gold_l == gold_l and next_gold_r == gold_r):
                                                should_exit = True
                                    elif [updated_gold_l, updated_gold_r, gold_label, gold_stop] == [left, right, label_idx, 1]:
                                        current_labels_at_labels[1] = 1
                                        c_idx += 1
                                        if c_idx < len(current_gold_labels):
                                            next_gold_l, next_gold_r, gold_label, gold_stop = current_gold_labels[c_idx]
                                            if not (next_gold_l == gold_l and next_gold_r == gold_r):
                                                should_exit = True
                                    current_labels_at_comb.append(current_labels_at_labels)
                            else:
                                current_labels_at_comb = [[0, 0] for _ in enumerate(uni_labels)]
                        current_labels_at_h.append(current_labels_at_comb)
                current_labels.append(current_labels_at_h)

            maximum_num_comb = max([len(current_labels[h_idx]) for h_idx in range(max_height)])
            for h_idx in range(max_height):
                current_labels[h_idx].extend([padded_labels_at_comb  for _ in range(maximum_num_comb - len(current_labels[h_idx]))])

            padded_height = max_height - len(feature.labels)
            label_height_mask = feature.label_height_mask + [0] * padded_height


            batch[i] = UniFeature(input_ids=np.asarray(input_ids),
                                attention_mask=np.asarray(attn_mask),
                                  token_type_ids=np.asarray(token_type_ids),
                                 variable_indexs_start=np.asarray(var_starts),
                                 variable_indexs_end=np.asarray(var_ends),
                                 num_variables=np.asarray(feature.num_variables),
                                 variable_index_mask=np.asarray(variable_index_mask),
                                 labels =np.asarray(current_labels), # binary_labels: (max_height,  num_combinations, num_op_labels, 2)
                                  label_height_mask=np.asarray(label_height_mask))
        results = UniFeature(*(default_collate(samples) for samples in zip(*batch)))
        return results


def get_transform_labels_from_batch_labels(batch):
    num_variables = batch.num_variables.cpu().numpy().tolist()  # batch_size
    max_num_variable = max(num_variables)
    batched_labels = batch.labels  ## (batch_size, max_height,  num_combinations, num_op_labels, 2)

    batch_size, max_height, num_combinations, _, _ = batched_labels.size()
    ## get max_variable per step
    maximum_prev_intermediate_for_h = [0]
    prev_intermediate_for_h = [[0] * batch_size]
    accumulate_max_prev_num_intermediate = [0]
    for h_idx in range(max_height):
        current_batch_labels = batched_labels[:, h_idx, :, :, :]
        judge = (current_batch_labels == 1).nonzero()  ## num_nonzerp x 3 (batch_idx, comb_idx, label_idx, stop_id)
        num_in_batch = torch.bincount(judge[:, 0], minlength=batch_size)  ## (num_in_b0, num_in_b1, ...)
        maximum_prev_intermediate_for_h.append(max(num_in_batch))
        prev_intermediate_for_h.append(num_in_batch)
        accumulate_max_prev_num_intermediate.append(
            maximum_prev_intermediate_for_h[-1] + accumulate_max_prev_num_intermediate[-1])
    all_transform_labels= []
    for b_idx, labels in enumerate(batched_labels):
        # num_var = num_variables[b_idx]
        transformed_labels = []
        padded_indexs = []
        latest_pad_idxs = []
        # if objs[b_idx]["id"] == '15336':
        #     print("sd")
        for h_idx, curr_comb_labels in enumerate(labels):
            # curr_comb_labels: num_combinations, num_op_labels, 2
            judge = (curr_comb_labels == 1).nonzero()  ## num_nonzerp x 3 (comb_idx, label_idx, stop_id)
            num_var_for_comb = accumulate_max_prev_num_intermediate[h_idx] + len(constant_values) + max_num_variable
            num_var_range = torch.arange(0, num_var_for_comb)
            combination = torch.combinations(num_var_range, r=2, with_replacement=True)  ##number_of_combinations x 2
            gold_comb = combination[judge[:, 0], :]  ## num_nonzero, 2
            gold_labels = judge[:, 1]  ## num_nonzero,
            gold_stop_ids = judge[:, 2]  ## num_nonzero,
            candidate_gold_labels = torch.cat([gold_comb, gold_labels.unsqueeze(-1), gold_stop_ids.unsqueeze(-1)],
                                              dim=-1)  # num_nonzero, 4.
            candidate_gold_labels = candidate_gold_labels.cpu().numpy().tolist()
            decoded_gold_labels = []

            for pid in range(len(padded_indexs)):
                padded_indexs[pid] += maximum_prev_intermediate_for_h[h_idx]
            for idx in latest_pad_idxs:
                if idx not in padded_indexs:
                    padded_indexs.append(idx)
            latest_pad_idxs = []
            for cand_l, cand_r, cand_label, cand_stop in candidate_gold_labels:
                decoded_l, decoded_r = cand_l, cand_r
                if cand_l >= accumulate_max_prev_num_intermediate[h_idx]:
                    decoded_l = decoded_l - len(padded_indexs)
                else:
                    for pad_idx in padded_indexs:
                        if decoded_l > pad_idx:
                            decoded_l -= 1
                if cand_r >= accumulate_max_prev_num_intermediate[h_idx]:
                    decoded_r = decoded_r - len(padded_indexs)
                else:
                    for pad_idx in padded_indexs:
                        if decoded_r > pad_idx:
                            decoded_r -= 1
                decoded_gold_labels.append([decoded_l, decoded_r, cand_label, cand_stop])
            decoded_gold_labels = sorted(decoded_gold_labels)
            curr_generated_intermediate = len(candidate_gold_labels)
            pad_num = maximum_prev_intermediate_for_h[h_idx + 1] - curr_generated_intermediate

            for i in range(maximum_prev_intermediate_for_h[h_idx + 1] - 1, curr_generated_intermediate - 1, -1):
                latest_pad_idxs.append(i)
            if len(decoded_gold_labels) > 0:
                transformed_labels.append(decoded_gold_labels)
            # assert sorted(insts[b_idx].labels) == sorted(transformed_labels)

            # print(insts[b_idx].labels)
            # print(transformed_labels)
            # assert insts[b_idx].labels == insts[b_idx].labels
        all_transform_labels.append(transformed_labels)
    return all_transform_labels

if __name__ == '__main__':
    from transformers import BertTokenizer
    from src.eval.utils import is_value_correct
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # dataset = UniversalDataset(file="../../data/complex/mwp_processed_train.json", tokenizer=tokenizer)
    constant2id = {"1": 0, "PI": 1}
    constant_values = [1.0, 3.14]
    add_replacement = True
    use_incremental_labeling = True
    # UniversalDataset(file="../../data/math23k/test23k_processed_nodup.json", tokenizer=tokenizer,
    #                  constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
    #                  use_incremental_labeling=use_incremental_labeling, add_new_token=False)
    # UniversalDataset(file="../../data/math23k/train23k_processed_nodup.json", tokenizer=tokenizer,
    #                  constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
    #                  use_incremental_labeling=use_incremental_labeling, add_new_token=False)
    # UniversalDataset(file="../../data/math23k/valid23k_processed_nodup.json", tokenizer=tokenizer,
    #                  constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
    #                  use_incremental_labeling=use_incremental_labeling, add_new_token=False)

    test_set = UniversalDataset(file="../../data/math23k/test23k_parallel_sorted.json", tokenizer=tokenizer,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, add_new_token=False)
    # train_set = UniversalDataset(file="../../data/math23k/train23k_parallel_sorted.json", tokenizer=tokenizer,
    #                  constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
    #                  use_incremental_labeling=use_incremental_labeling, add_new_token=False, number=-1)
    valid_set = UniversalDataset(file="../../data/math23k/valid23k_parallel_sorted.json", tokenizer=tokenizer,
                                 constant2id=constant2id, constant_values=constant_values,
                                 add_replacement=add_replacement,
                                 use_incremental_labeling=use_incremental_labeling, add_new_token=False, number=-1)
    # valid_set = UniversalDataset(file="../../data/math23k/debug_parallel.json", tokenizer=tokenizer,
    #                  constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
    #                  use_incremental_labeling=use_incremental_labeling, add_new_token=False, number=-1)
    from torch.utils.data import DataLoader
    loader  = DataLoader(test_set, batch_size=30, collate_fn=test_set.collate_parallel)
    # exit(0)
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        ## TODO: check retrieve the label from loader and get back the results.
        pass
        # continue
        # print(batch.labels[0])
        # if batch_idx == 191:
        #     print(batch_idx)
        insts = loader.dataset._features[batch_idx*(loader.batch_size):(batch_idx+1)*(loader.batch_size)]
        objs = loader.dataset.insts[batch_idx*(loader.batch_size):(batch_idx+1)*(loader.batch_size)]
        all_transform_labels = get_transform_labels_from_batch_labels(batch)

        for b_idx, transformed_labels in enumerate(all_transform_labels):
            checker = is_value_correct(transformed_labels, insts[b_idx].labels, objs[b_idx]["num_list"], num_constant=2,
                             constant_values=constant_values,
                             consider_multiple_m0=True, use_parallel_equations=True)
            # print(checker)
            try:
                assert checker[0]
            except:
                print(insts[b_idx].labels)
                print(transformed_labels)
                print(objs[b_idx]["id"], batch_idx, b_idx)
                print("error")