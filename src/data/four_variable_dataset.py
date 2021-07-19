from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizerFast, MBartTokenizerFast
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import numpy as np
from src.utils import read_data
from src.data.math_dataset import labels, Feature



class FourVariableDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1,
                 test_strings:List[str] = None) -> None:
        self.tokenizer = tokenizer
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            variables = obj["variables"]
            ans_variable = obj["ans_variable"]
            all_generated_m0 = obj["all_generated_m0"]
            equation = obj["equation"]
            assert equation.startswith("x=")
            v_name2variables = {var["var_name"]: var for var in variables}

            all_variables = variables + [ans_variable]
            all_variables_sorted = sorted(all_variables, key=lambda obj: obj["order_in_text"])
            v_name2idx = {var["var_name"]: idx for idx, var in enumerate(all_variables_sorted)}
            all_ids_diff_m0 = []
            all_sent_starts = []
            all_sent_ends = []
            all_attn_mask = []
            gold_m0 = obj["m0"]
            all_label_ids = []
            label_sum = 0
            assert len(all_generated_m0) == 18
            for m0_idx, generated_m0_obj in enumerate(all_generated_m0):
                all_strings = [var["concat_text"] for var in all_variables_sorted]
                comb = generated_m0_obj["comb"]
                left_idx_str, right_idx_str = comb.split(" ")
                all_var_names = [1, 2, 3]
                left_idx, right_idx = int(left_idx_str), int(right_idx_str)
                all_var_names.remove(left_idx)
                all_var_names.remove(right_idx)
                m0_on_smaller = False
                if left_idx + right_idx <= all_var_names[0]:
                    m0_on_smaller = True
                operator = generated_m0_obj["operator"].replace("x", "*")
                m0_cand = "v" + left_idx_str + operator + "v" + right_idx_str if not operator.endswith("_rev") else "v" + right_idx_str + operator[0] + "v" + left_idx_str
                curr_m0_label_id = 0
                if m0_cand == gold_m0:
                    curr_m0_label_id = 1
                label_ids = [0] * len(labels)
                if curr_m0_label_id == 1:
                    replaced_equation = equation.replace(f"({m0_cand})", f"m0")
                    assert "m0" in replaced_equation
                    if "+" in replaced_equation:
                        label_ids[labels.index("+")] = 1
                    elif "*" in replaced_equation:
                        label_ids[labels.index("+")] = 1
                    elif "-" in replaced_equation:
                        if replaced_equation.endswith("m0"):
                            if m0_on_smaller:
                                label_ids[labels.index("-_rev")] = 1
                            else:
                                label_ids[labels.index("-")] = 1
                        else:
                            if m0_on_smaller:
                                label_ids[labels.index("-")] = 1
                            else:
                                label_ids[labels.index("-_rev")] = 1
                    elif "/" in replaced_equation:
                        if replaced_equation.endswith("m0"):
                            if m0_on_smaller:
                                label_ids[labels.index("/_rev")] = 1
                            else:
                                label_ids[labels.index("/")] = 1
                        else:
                            if m0_on_smaller:
                                label_ids[labels.index("/")] = 1
                            else:
                                label_ids[labels.index("/_rev")] = 1
                label_sum += sum(label_ids)
                if label_sum > 1:
                    print("some error")
                all_label_ids.append(label_ids)
                gen_m0_string = generated_m0_obj["generated_m0"]
                left_var = v_name2variables["v" + left_idx_str]
                right_var = v_name2variables["v" + right_idx_str]
                right_most_idx = v_name2idx["v" + right_idx_str]
                left_most_idx = v_name2idx["v" + left_idx_str]
                assert right_most_idx > left_most_idx
                all_strings.insert(right_most_idx+1, gen_m0_string)
                assert len(all_strings) == 5
                res = tokenizer.batch_encode_plus(all_strings, return_attention_mask=False, add_special_tokens=False)
                sent_starts = []
                sent_ends = []
                all_ids = [tokenizer.cls_token_id]
                start = len(all_ids)
                for k, ids in enumerate(res["input_ids"]):
                    if k != right_most_idx+1 and (k == left_most_idx or k == right_most_idx):
                        pass
                    else:
                        sent_starts.append(start)
                        sent_ends.append(start + len(ids))
                    all_ids.extend(ids)
                    if k != len(res["input_ids"]) - 1:
                        all_ids.append(tokenizer.convert_tokens_to_ids(['，'])[0])
                    else:
                        all_ids.append(tokenizer.convert_tokens_to_ids(['？'])[0])
                    start = len(all_ids)
                all_ids.append(tokenizer.sep_token_id)
                assert len(sent_starts) == 3
                attn_mask = [1] * len(all_ids)
                all_ids_diff_m0.append(all_ids)
                all_sent_starts.append(sent_starts)
                all_sent_ends.append(sent_ends)
                all_attn_mask.append(attn_mask)
            assert label_sum == 1
            self._features.append(Feature(dataset='4_var', input_ids=all_ids_diff_m0,
                                          attention_mask=all_attn_mask,
                                          sent_starts=all_sent_starts,
                                          sent_ends=all_sent_ends,
                                          label_id=all_label_ids))

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> Feature:
        return self._features[idx]

    def collate_function(self, batch: List[Feature]):
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
            batch[i] = Feature(dataset=feature.dataset, input_ids=all_padded_ids,
                              attention_mask=np.asarray(all_padded_attn_mask),
                              sent_starts=np.asarray(feature.sent_starts),
                              sent_ends=np.asarray(feature.sent_ends),
                              label_id=np.asarray(feature.label_id))
        results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return results


if __name__ == '__main__':
    tokenizer = MBartTokenizerFast.from_pretrained('facebook/mbart-large-cc25')
    dataset = FourVariableDataset(file='../../data/all_generated_1.0.json', tokenizer=tokenizer, number=-1)
    # from torch.utils.data import DataLoader
    #
    # loader = DataLoader(dataset, batch_size=3,shuffle=True,collate_fn=dataset.collate_function)
    # for batch in loader:
    #     pass
        # print(batch)
