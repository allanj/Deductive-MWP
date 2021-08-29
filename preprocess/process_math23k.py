

from src.utils import read_data, write_data
from typing import List, Dict
import re
from tqdm import tqdm
from collections import Counter

def have_constant_square(target_template: List) -> bool:
    for val in target_template:
        if val.strip() == "1" or val.strip() == "pi" or val.strip() == "PI" or val.strip() == "^":
            return True
    return False

def count_variable(target_template: List) -> int:
    num_vars = set()
    for val in target_template:
        if val.strip().startswith("temp_"):
            num_vars.add(val.strip())
    return len(num_vars)

def have_multiple_m0(target_template: List):
    target_string = ' '.join(target_template)
    target_string = target_string.replace("()", "").replace("( )", "")
    target_string = re.sub(r"\(.*\)", "temp_m", target_string)
    target_template = target_string.split()
    high_priority_symbol_pos = []
    for idx, val in enumerate(target_template):
        if val in {"*", "/"}:
            high_priority_symbol_pos.append(idx)
    for prev, next in zip(high_priority_symbol_pos[:-1], high_priority_symbol_pos[1:]):
        if next - prev != 2:
            return True
    return False

def get_labels(target_norm_post_template: List):
    assert target_norm_post_template[:2] == ["x", "="]
    stack = []
    pointer = 2
    labels = []
    both_m = False
    while pointer != len(target_norm_post_template):
        stack.append(target_norm_post_template[pointer])
        if stack[-1] in {'+', '-', '*', '/', '^'}:
            if len(stack[-3:]) == 3:
                if stack[-3].startswith("m") and stack[-2].startswith("m"):
                    both_m = True
                labels.append([stack[-3], stack[-2], stack[-1]])
                stack.pop()
                stack.pop()
                stack.pop()
                stack.append(f"m_{len(labels)}")
        pointer += 1
    for i, (left, right, op) in enumerate(labels):
        if i == 0:
            left = left[-1:]
            right = right[-1:]
            if left > right and op in {'-', '/', '^'}:
                labels[i] = [right, left, op+"_rev"]
            else:
                labels[i] = [left, right, op]
        else:
            if left.startswith("m"):
                right = right[-1:]
                labels[i] = ["#", right, op]
            else:
                left = left[-1:]
                labels[i] = ["#", left, op+"_rev"] if op in {'-', '/', '^'} else ["#", left, op]

    return labels, both_m


def process_obj(obj: Dict):
    target_template = [val.strip() for val in obj["target_template"]]

    labels, have_both_m = get_labels(obj["target_norm_post_template"])
    type_str = "legal"

    if count_variable(target_template) > 4:
        type_str = "variable more than 4"
        return type_str, labels

    if have_constant_square(target_template):
        type_str = "have constant"
        return type_str, labels

    if have_multiple_m0(target_template):
        type_str = "have mutiple m0"
        return type_str, labels

    if have_both_m:
        type_str = "have both m0, m1"
        return type_str, labels
    return type_str, labels

def main():
    for in_file in ["train23k_processed.json", "valid23k_processed.json", "test23k_processed.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/math23k/{in_file}"
        out_file = in_file.split(".json")[0] + "_labeled.json"
        data = read_data(in_file)
        count = Counter()
        for obj in tqdm(data, desc="processing data", total=len(data)):
            type_str, labels = process_obj(obj)
            count[type_str] += 1
            obj["type_str"] = type_str
            obj["equation_layer"] = labels
        write_data(file=out_file, data = data)

        for key in count:
            print(f"{key}, valid number: {count[key]}, total: {len(data)}, %: {count[key] * 1.0 / len(data) * 100:.2f}")

if __name__ == '__main__':
    # text = "a () * c"
    # print(re.sub(r"\(.*\)", "temp_m", text))
    main()
    # print(breakit('(((a+b)+a)+c)'))