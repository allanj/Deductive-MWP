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

UniFeature = collections.namedtuple('UniFeature', 'input_ids attention_mask variable_indexs_start variable_indexs_end num_variables variable_index_mask labels')
UniFeature.__new__.__defaults__ = (None,) * 7

class UniversalDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1) -> None:
        self.tokenizer = tokenizer
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            if not (obj['legal'] and obj['num_steps'] <= 2):
                continue
            ## equation preprocessing
            mapped_equation = obj["mapped_equation"]
            for k in range(ord('a'), ord('a') + 26):
                mapped_equation = mapped_equation.replace(f"( temp_{chr(k)} )", f"temp_{chr(k)}")
            pattern = r"\( ?\( ?temp_\w [\+\-\*\/] temp_\w ?\) ?\)"
            if len(re.findall(pattern, mapped_equation)) > 0:
                mapped_equation = mapped_equation.replace("( ( ", "( ").replace(") ) ", ") ")
            ### end equation preprocess
            mapped_text = obj["mapped_text"].split()
            input_text = ""
            for word in mapped_text:
                if word.startswith("temp_"):
                    input_text += " <quant> "
                elif word == ",":
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
                if token == "<" and tokens[k:] == ['<', 'q', '##uan', '##t', '>']:
                    var_starts.append(k)
                    var_ends.append(k+5)
            num_variable = len(var_starts)
            var_mask = [1] * num_variable
            labels = self.get_label_ids(mapped_equation)
            self._features.append(
                UniFeature(input_ids=input_ids,
                           attention_mask=attention_mask,
                           variable_indexs_start=var_starts,
                           variable_indexs_end=var_ends,
                           num_variables=num_variable,
                           variable_index_mask=var_mask,
                           labels = labels)
            )

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> UniFeature:
        return self._features[idx]




    def get_label_ids(self, mapped_equation) -> Union[List[List[int]], None]:
        # in this data, only have one or zero bracket
        assert len(mapped_equation) - len(mapped_equation.replace("(", "")) <= 1
        assert mapped_equation.startswith("x=")
        mapped_equation = ' '.join(mapped_equation[2:].split(' '))
        all_labels = []
        if  len(mapped_equation) - len(mapped_equation.replace("(", "")) == 1:
            var_and_ops = mapped_equation.split(" ")
            left_bracket_index = var_and_ops.index("(")
            right_bracket_index = var_and_ops.index(")")
            ## height = 0
            op = var_and_ops[left_bracket_index + 2]
            left_var_index = ord(var_and_ops[left_bracket_index + 2 - 1][-1]) - ord('a')
            right_var_index = ord(var_and_ops[left_bracket_index + 2 + 1][-1]) - ord('a')
            actual_op_id = uni_labels.index(op) if left_var_index < right_var_index else uni_labels.index(op + "_rev")
            labels = [left_var_index, right_var_index, actual_op_id]
            all_labels.append(labels)
            if left_bracket_index != 0

        else:
            var_and_ops = mapped_equation.split(" ")
            h = 0
            for i in range(1, len(var_and_ops), 2):
                op = var_and_ops[i]
                assert op in ["+", "-", "*", "/"]
                left_var_index = ord(var_and_ops[i-1][-1]) - ord('a')if i == 1 else -1
                right_var_index = ord(var_and_ops[i+1][-1]) - ord('a')
                if left_var_index == right_var_index:
                    return None
                actual_op_id = uni_labels.index(op) if left_var_index < right_var_index else uni_labels.index(op + "_rev")
                labels = [left_var_index, right_var_index, actual_op_id]
                all_labels.append(labels)
                h += 1
        return all_labels



    def collate_function(self, batch: List[UniFeature]):
        """
        This one is wrong, I haven't modified yet.


        :param batch:
        :return:
        """


        max_wordpiece_length = max([len(ids)  for feature in batch for ids in feature.input_ids])
        for i, feature in enumerate(batch):
            all_padded_ids = []
            all_padded_attn_mask = []
            for ids,attn_mask in zip(feature.input_ids, feature.attention_mask):
                padding_length = max_wordpiece_length - len(ids)
                padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                padded_attn_mask = attn_mask + [0] * padding_length
                all_padded_ids.append(padded_ids)
                all_padded_attn_mask.append(padded_attn_mask)
            all_padded_ids = np.asarray(all_padded_ids)
            assert np.asarray(feature.label_id).sum()==1
            batch[i] = UniFeature(dataset=feature.dataset, input_ids=all_padded_ids,
                              attention_mask=np.asarray(all_padded_attn_mask),
                                 sent_starts=np.asarray(feature.sent_starts),
                                 sent_ends=np.asarray(feature.sent_ends),
                                 m0_sent_starts=np.asarray(feature.m0_sent_starts),
                                 m0_sent_ends=np.asarray(feature.m0_sent_ends),
                                 m0_operator_ids =np.asarray(feature.m0_operator_ids),
                              label_id=np.asarray(feature.label_id))
        results = UniFeature(*(default_collate(samples) for samples in zip(*batch)))
        return results


if __name__ == '__main__':
    text = "x= temp_b / ( ( temp_b * temp_c ) )"
    pattern = r"\( ?\( ?temp_\w [\+\-\*\/] temp_\w ?\) ?\)"
    import re
    print(re.findall(pattern, text))
