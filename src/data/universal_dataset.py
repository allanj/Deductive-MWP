from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizerFast, MBartTokenizerFast
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import numpy as np
from src.utils import read_data, write_data
import collections
import re
from src.eval.utils import compute_value, compute_value_for_incremental_equations
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
                 use_incremental_labeling: bool = False) -> None:
        self.tokenizer = tokenizer
        self.constant2id = constant2id
        self.constant_values = constant_values
        self.constant_num = len(self.constant2id) if self.constant2id else 0
        self.use_incremental_labeling = use_incremental_labeling
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

            if self.use_incremental_labeling:
                labels = self.get_label_ids_incremental(obj["equation_layer"], add_replacement=add_replacement)
            else:
                labels = self.get_label_ids_updated(obj["equation_layer"], add_replacement=add_replacement)

            if not labels:
                num_has_same_var_m0 += 1
                obj['type_str'] = "illegal"
                continue
            # compute_value(labels, obj["num_list"])

            if len(labels) > 10:
                numbert_instances_filtered += 1
                continue

            if isinstance(labels, str):
                num_index_error += 1
                obj['type_str'] = "illegal"
                continue
            if self.use_incremental_labeling:
                # if obj["id"] == '163':
                #     print("sss")
                res = compute_value_for_incremental_equations(labels, obj["num_list"], self.constant_num, constant_values=self.constant_values)
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

            if left_var != "#" and (not left_var.startswith("m")):
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
            if (not left_var.startswith("m")):
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
            if (not right_var.startswith("m")):
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
                        if (op in ["+", "*"]):
                            op_idx = uni_labels.index(op)
                            label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                        else:
                            # assert not op.endswith("_rev")
                            op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                            label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
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


if __name__ == '__main__':
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # dataset = UniversalDataset(file="../../data/complex/mwp_processed_train.json", tokenizer=tokenizer)
    constant2id = {"1": 0, "PI": 1}
    constant_values = [1.0, 3.14]
    add_replacement = True
    use_incremental_labeling = True
    UniversalDataset(file="../../data/math23k/test23k_processed_all.json", tokenizer=tokenizer,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling)
    UniversalDataset(file="../../data/math23k/train23k_processed_all.json", tokenizer=tokenizer,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling)
    UniversalDataset(file="../../data/math23k/valid23k_processed_all.json", tokenizer=tokenizer,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling)

    # UniversalDataset(file="../../data/math23k/test23k_processed_labeled.json", tokenizer=tokenizer)
    # UniversalDataset(file="../../data/math23k/train23k_processed_labeled.json", tokenizer=tokenizer)
    # UniversalDataset(file="../../data/math23k/valid23k_processed_labeled.json", tokenizer=tokenizer)
