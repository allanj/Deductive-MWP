from torch.utils.data import Dataset
from typing import Dict, List, Union, Tuple
import json
from transformers import T5TokenizerFast, PreTrainedTokenizerFast, MBartTokenizerFast
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

Feature = collections.namedtuple('Feature', 'input_ids attention_mask labels')
Feature.__new__.__defaults__ = (None,) * 5


class Seq2SeqDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1) -> None:
        self.tokenizer = tokenizer
        if "math23k" in file:
            self.read_math23k(file, tokenizer, number)
        else:
            data = read_data(file=file)
            if number > 0:
                data = data[:number]
            # ## tokenization
            self._features = []
            for obj in tqdm(data, desc='Tokenization', total=len(data)):
                if not (obj['legal'] and obj['num_steps'] <= 2):
                    continue
                assert obj["posted_equation"].startswith("x =")
                post_equation = obj["posted_equation"].replace("x =", "").strip().replace("temp_", "")
                post_equation = ' '.join(post_equation.split())
                words = obj["mapped_text"].split()
                input_text = ""
                for word in words:
                    if word.startswith("temp_"):
                        input_text += word[-1:]
                    elif word == ",":
                        input_text += word + " "
                    else:
                        input_text += word
                res = tokenizer.encode_plus(input_text, add_special_tokens=True, return_attention_mask=True)
                input_ids = res["input_ids"]
                attention_mask = res["attention_mask"]

                label_ids = tokenizer.encode_plus(post_equation, add_special_tokens=True, return_attention_mask=False)["input_ids"]
                self._features.append(Feature(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              labels= label_ids))
            print(f"length of data: {len(self._features)}")

    def read_math23k(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1) -> None:
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            target_template = ' '.join(obj["target_norm_post_template"])
            assert target_template.startswith("x =")
            post_equation = target_template.replace("x =", "").strip().replace("temp_", "")
            post_equation = ' '.join(post_equation.split())
            words = obj["text"].split()
            input_text = ""
            for word in words:
                if word.startswith("temp_"):
                    input_text += word[-1:]
                elif word == ",":
                    input_text += word + " "
                else:
                    input_text += word
            res = tokenizer.encode_plus(input_text, add_special_tokens=True, return_attention_mask=True)
            input_ids = res["input_ids"]
            attention_mask = res["attention_mask"]

            label_ids = tokenizer.encode_plus(post_equation, add_special_tokens=True, return_attention_mask=False)[
                "input_ids"]
            self._features.append(Feature(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          labels=label_ids))
        print(f"length of data: {len(self._features)}")

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> Feature:
        return self._features[idx]

    def collate_function(self, batch: List[Feature]):
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        max_answer_length = max([len(feature.labels) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            pad_label_length = max_answer_length - len(feature.labels)
            batch[i] = Feature(input_ids=np.asarray(feature.input_ids + [self.tokenizer.pad_token_id] * padding_length),
                              attention_mask=np.asarray(feature.attention_mask + [0] * padding_length),
                              labels=np.asarray(feature.labels + [self.tokenizer.pad_token_id] * pad_label_length))
        results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return results


if __name__ == '__main__':
    tokenizer = MBartTokenizerFast.from_pretrained('facebook/mbart-large-cc25')
    dataset = Seq2SeqDataset(file='../../data/math23k/train23k_processed.json', tokenizer=tokenizer, number=100)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=3,shuffle=True,collate_fn=dataset.collate_function)
    for batch in loader:
        pass
        # print(batch)
