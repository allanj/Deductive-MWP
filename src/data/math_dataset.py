from torch.utils.data import Dataset
from typing import Dict, List, Union, Tuple
import json
from transformers import PreTrainedTokenizerFast, RobertaTokenizerFast
import re
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import numpy as np
from src.utils import read_data
from dataclasses import dataclass
import collections
import math
from src.data.utils import fv_labels
from collections import Counter

Feature = collections.namedtuple('Feature', 'input_ids attention_mask sent_starts sent_ends label_id')
Feature.__new__.__defaults__ = (None,) * 5

from sympy import symbols, solve
# labels = [
#     '0 1 2 +', # v0 + v1 = v2
#     '0 2 1 +',
#     '1 2 0 +',
#     '0 1 2 *',
#     '0 2 1 *',
#     '1 2 0 *',
# ]

labels = [
    '+','-', '-_rev', '*', '/', '/_rev'
]

# pretty_labels = [
#     'V0 + V1 = V2', # v0 + v1 = v2
#     'V0 + V2 = V1',
#     'V1 + V2 = V0',
#     'V0 * V1 = V2',
#     'V0 * V2 = V1',
#     'V1 * V2 = V0'
# ]

pretty_labels = [
    'x=v1+v2', # v0 + v1 = v2
    'x=v1-v2', # v0 + v1 = v2
'x=v2-v1', # v0 + v1 = v2
'x=v1*v2', # v0 + v1 = v2
'x=v1/v2', # v0 + v1 = v2
'x=v2/v1', # v0 + v1 = v2
]



def measure_two_expr(gold_expr_str, pred_expr_str, nums, ans_idx):
    gold_vals = gold_expr_str.split()
    pred_vals = pred_expr_str.split()
    gold_expr_vals = [nums[idx] if idx != ans_idx else symbols('x') for idx in
                      [int(v) for v in gold_vals[:3]]]
    pred_expr_vals = [nums[idx] if idx != ans_idx else symbols('x') for idx in
                      [int(v) for v in pred_vals[:3]]]
    if gold_vals[-1] == "+":
        gold_expr = gold_expr_vals[0] + gold_expr_vals[1] - gold_expr_vals[2]
    elif gold_vals[-1] == "*":
        gold_expr = gold_expr_vals[0] * gold_expr_vals[1] / gold_expr_vals[2] - 1
    if pred_vals[-1] == "+":
        pred_expr = pred_expr_vals[0] + pred_expr_vals[1] - pred_expr_vals[2]
    elif pred_vals[-1] == "*":
        pred_expr = pred_expr_vals[0] * pred_expr_vals[1] / pred_expr_vals[2] - 1
    gold_sol = solve(gold_expr)
    pred_sol = solve(pred_expr)
    if len(gold_sol) > 0 and len(pred_sol) > 0 and math.fabs((gold_sol[0] - pred_sol[0])) < 1e-5:
        return True
    return False

class QuestionDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1,
                 use_four_variables: bool = True,
                 test_strings:List[str] = None,
                 use_binary: bool  =False) -> None:
        self.tokenizer = tokenizer
        self.use_binary = use_binary
        if test_strings:
            self._features = []
            for question in test_strings:
                question = question.replace(",", "，")
                segments = question.split("，")
                id_lists = tokenizer.batch_encode_plus(segments, add_special_tokens=False, return_attention_mask=False)[
                    "input_ids"]
                all_ids = [tokenizer.cls_token_id]
                start = len(all_ids)
                sent_starts = []
                sent_ends = []
                for k, ids in enumerate(id_lists):
                    sent_starts.append(start)
                    sent_ends.append(start + len(ids))
                    all_ids.extend(ids)
                    if k != len(id_lists) - 1:
                        all_ids.append(tokenizer.convert_tokens_to_ids(['，'])[0])
                    else:
                        all_ids.append(tokenizer.convert_tokens_to_ids(['？'])[0])
                    start = len(all_ids)
                attn_mask = [1] * len(all_ids)
                label_id = -1
                self._features.append(Feature(input_ids=all_ids,
                                              attention_mask=attn_mask,
                                              sent_starts=sent_starts,
                                              sent_ends=sent_ends,
                                              label_id=label_id))
        else:
            data = read_data(file=file)
            if number > 0:
                data = data[:number]
            num_3_variables = 0
            insts = []
            num_without_x = 0
            label2count = Counter()
            for obj in tqdm(data, total=len(data), desc="Reading data"):
                if not use_four_variables and len(obj["variables"]) != 3:
                    continue
                found_no_x = True
                for var in obj["variables"]:
                    if var[0] == 'x':
                        found_no_x = False
                        break
                if found_no_x:
                    num_without_x += 1
                    continue
                num_3_variables += 1
                insts.append(obj)
            self.insts = insts
            # ## tokenization
            print(f"number of instances have 3 variables: {num_3_variables} over total: {len(data)}, num instanes (removed) without x variable: {num_without_x}")
            self._features = []
            for inst in tqdm(insts, desc='Tokenization', total=len(insts)):
                sents = []
                sent_starts = []
                sent_ends = []
                nums = []
                ans_idx = -1
                ans = -1
                for var_idx, variable in enumerate(inst["variables"]):
                    words = variable[2]
                    curr_num = variable[1]
                    if variable[0] == 'x':
                        ans_idx = var_idx
                        ans = curr_num
                    if int(curr_num) == curr_num:
                        curr_num = int(curr_num)
                    nums.append(curr_num)
                    for i, word in enumerate(words):
                        if word.startswith("<") and word.endswith(">"):
                            if variable[0] == 'x':
                                words[i] = "多少"
                            else:
                                words[i] = str(curr_num)
                    sents.append(''.join(words))
                inst["nums"] = nums
                inst["ans"] = ans
                inst["ans_idx"] = ans_idx
                assert ans_idx >= 0
                id_lists = tokenizer.batch_encode_plus(sents,add_special_tokens=False, return_attention_mask=False)["input_ids"]
                all_ids = [tokenizer.cls_token_id]
                start = len(all_ids)
                for k, ids in enumerate(id_lists):
                    sent_starts.append(start)
                    sent_ends.append(start + len(ids))
                    all_ids.extend(ids)
                    if k != len(id_lists) - 1:
                        all_ids.append(tokenizer.convert_tokens_to_ids(['，'])[0])
                    else:
                        all_ids.append(tokenizer.convert_tokens_to_ids(['？'])[0])
                    start = len(all_ids)
                all_ids.append(tokenizer.sep_token_id)
                attn_mask = [1]*len(all_ids)
                # label_id = -1
                equation = inst["equation"]
                vals = equation.split(" ")
                gold_operator = vals[3]
                if vals[2].strip() == "v1":
                    label_id = labels.index(gold_operator)
                else:
                    label_id = labels.index(gold_operator + "_rev")
                # curr_labels = labels if len(nums) == 3 else fv_labels
                # for lid, possible_label in enumerate(curr_labels):
                #     vals = possible_label.split()
                #     indices = [int(v) for v in vals[:3]]
                #     if all([element != ans_idx for element in indices]): ## no x variable
                #         continue
                #     expr_vals = [nums[idx] if idx != ans_idx else symbols('x') for idx in indices]
                #     expr = None
                #     if vals[-1] == "+":
                #         expr = expr_vals[0] + expr_vals[1] - expr_vals[2]
                #     elif vals[-1] == "*":
                #         expr = expr_vals[0] * expr_vals[1] / expr_vals[2] - 1
                #     sol = solve(expr)
                #     if len(sol) > 0 and math.fabs((sol[0] - ans)) < 1e-5:
                #         label_id = lid
                #         label2count[label_id] +=1
                #         break ## because only one label should be available for three variables
                if use_binary:
                    cands = [0] * len(labels)
                    assert label_id != -1
                    cands[label_id] = 1
                    label_id = cands
                self._features.append(Feature(input_ids=all_ids,
                                              attention_mask=attn_mask,
                                              sent_starts=sent_starts,
                                              sent_ends=sent_ends,
                                              label_id=label_id))
            print(label2count)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> Feature:
        return self._features[idx]

    def collate_function(self, batch: List[Feature]):
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        max_num_sents = max([len(feature.sent_starts) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            mask = feature.attention_mask + [0] * padding_length
            sent_starts = feature.sent_starts + [0] * (max_num_sents - len(feature.sent_starts))
            sent_ends = feature.sent_ends + [0] * (max_num_sents - len(feature.sent_ends))
            label_id = feature.label_id if not self.use_binary else np.asarray(feature.label_id)
            batch[i] = Feature(input_ids=np.asarray(input_ids),
                               attention_mask=np.asarray(mask),
                               sent_starts=np.asarray(sent_starts),
                               sent_ends=np.asarray(sent_ends), label_id=label_id)
        # If `is_mtl` is true, the first one is the dataset name
        results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return results


if __name__ == '__main__':
    # tokenizer = RobertaTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # dataset = QuestionDataset(file='../../data/simple_cases.json', tokenizer=tokenizer, number=-1)
    # from torch.utils.data import DataLoader
    #
    # loader = DataLoader(dataset, batch_size=3,shuffle=True,collate_fn=dataset.collate_function)
    # for batch in loader:
    #     pass
    #     # print(batch)

    numbers = re.findall(r"\d+", "我们班有100个人")
    print(numbers)