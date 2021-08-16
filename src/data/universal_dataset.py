from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizerFast, MBartTokenizerFast
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import numpy as np
from src.utils import read_data
import torch
import collections
import re

"""
not finished yet
"""
uni_labels = [
    '+','-', '-_rev', '*', '/', '/_rev'
]

UniFeature = collections.namedtuple('UniFeature', 'input_ids attention_mask token_type_ids variable_indexs_start variable_indexs_end num_variables variable_index_mask labels')
UniFeature.__new__.__defaults__ = (None,) * 7

class UniversalDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1, remove_repeated: bool = True) -> None:
        self.tokenizer = tokenizer
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        num_has_same_var_m0 = 0
        max_num_steps = 0
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            # if not (obj['legal'] and obj['num_steps'] <= 2):
            #     continue
            if not obj['legal']:
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
            labels = self.get_label_ids(obj["equation_layer"], remove_repeated=remove_repeated)
            if not labels:
                num_has_same_var_m0 += 1
                continue
            max_num_steps = max(max_num_steps, len(labels))
            self._features.append(
                UniFeature(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids = [0] * len(input_ids),
                           variable_indexs_start=var_starts,
                           variable_indexs_end=var_ends,
                           num_variables=num_variable,
                           variable_index_mask=var_mask,
                           labels = labels)
            )
        print(f"number of instances that have same variable in m0: {num_has_same_var_m0}, total number instances: {len(self._features)},"
              f"max num steps: {max_num_steps}")

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> UniFeature:
        return self._features[idx]




    def get_label_ids(self, equation_layers: List, remove_repeated: bool) -> Union[List[List[int]], None]:
        # in this data, only have one or zero bracket
        label_ids = []
        for layer in equation_layers:
            left_var, right_var, op = layer
            if left_var == right_var and remove_repeated:
                return None
            left_var_idx = ord(left_var) - ord('a') if left_var != "#" else -1
            right_var_idx = ord(right_var) - ord('a')
            if left_var_idx < right_var_idx:
                op_idx = uni_labels.index(op)
                label_ids.append([left_var_idx, right_var_idx, op_idx])
            else:
                # left > right
                if op in ["+", "*"]:
                    op_idx = uni_labels.index(op)
                    label_ids.append([right_var_idx, left_var_idx, op_idx])
                else:
                    assert not op.endswith("_rev")
                    op_idx = uni_labels.index(op + "_rev")
                    label_ids.append([right_var_idx, left_var_idx, op_idx])
        return label_ids



    def collate_function(self, batch: List[UniFeature]):

        max_wordpiece_length = max([len(feature.input_ids)  for feature in batch])
        max_num_variable = max([feature.num_variables  for feature in batch])
        max_height = max([len(feature.labels) for feature in batch])
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
            labels = feature.labels + [[-1, max_num_variable, 0]]* padded_height

            batch[i] = UniFeature(input_ids=np.asarray(input_ids),
                                attention_mask=np.asarray(attn_mask),
                                  token_type_ids=np.asarray(token_type_ids),
                                 variable_indexs_start=np.asarray(var_starts),
                                 variable_indexs_end=np.asarray(var_ends),
                                 num_variables=np.asarray(feature.num_variables),
                                 variable_index_mask=np.asarray(variable_index_mask),
                                 labels =np.asarray(labels))
        results = UniFeature(*(default_collate(samples) for samples in zip(*batch)))
        return results


if __name__ == '__main__':
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    dataset = UniversalDataset(file="../../data/complex/mwp_processed_train.json", tokenizer=tokenizer)
