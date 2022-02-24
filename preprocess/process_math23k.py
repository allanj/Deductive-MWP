import random

from src.utils import read_data, write_data
from typing import List, Dict
import re
from tqdm import tqdm
from collections import Counter

def have_constant(target_template: List) -> bool:
    for val in target_template:
        if val.strip() == "1":
        # if val.strip() == "pi" or val.strip() == "PI":
            return True
    return False

def have_pi(target_template: List) -> bool:
    for val in target_template:
        if val.strip() == "PI":
        # if val.strip() == "pi" or val.strip() == "PI":
            return True
    return False

def have_square(target_template: List) -> bool:
    for val in target_template:
        if val.strip() == "^":
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

def check_in_labels(current_tuple, labels):
    if current_tuple in labels:
        return current_tuple
    if current_tuple[-1] in {'+', '*'} and [current_tuple[1], current_tuple[0], current_tuple[-1]] in labels:
        return [current_tuple[1], current_tuple[0], current_tuple[-1]]
    return None

def get_labels(target_norm_post_template: List, target_template: List, remove_duplicate: bool = False):
    assert target_norm_post_template[:2] == ["x", "="]
    if len(target_norm_post_template) == 3:
        assert target_norm_post_template[2].startswith("temp_")
        target_norm_post_template.append("1")
        target_norm_post_template.append("*")
    stack = []
    pointer = 2
    labels = []
    both_m = False
    eq_2_m = {}
    got_duplicate = False
    while pointer != len(target_norm_post_template):
        stack.append(target_norm_post_template[pointer])
        if stack[-1] in {'+', '-', '*', '/', '^'}:
            if len(stack[-3:]) == 3:
                if stack[-3].startswith("m_") and stack[-2].startswith("m_"):
                    both_m = True
                if remove_duplicate:
                    checker = check_in_labels([stack[-3], stack[-2], stack[-1]], labels)
                    if checker:
                        got_duplicate = True
                        m_string = eq_2_m[' '.join(checker)]
                    else:
                        labels.append([stack[-3], stack[-2], stack[-1]])
                        m_string = f"m_{len(labels)}"
                        eq_2_m[' '.join([stack[-3], stack[-2], stack[-1]])] = m_string
                else:
                    labels.append([stack[-3], stack[-2], stack[-1]])
                    m_string = f"m_{len(labels)}"
                stack.pop()
                stack.pop()
                stack.pop()
                stack.append(m_string)
        pointer += 1
    for i, (left, right, op) in enumerate(labels):
        # left = left[-1:] if left.startswith("temp_") else left
        # right = right[-1:] if right.startswith("temp_") else right
        if left.startswith("m_") or right.startswith("m_"):
            if left.startswith("m_") and right.startswith("m_"):
                left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
            elif right.startswith("m_"):
                modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                labels[i] = [right, left, modified_op]
        else:
            if left.startswith("temp_") or right.startswith("temp_"):
                if left.startswith("temp_") and right.startswith("temp_"):
                    left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                    modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                    labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
                elif right.startswith("temp_"):
                    modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                    labels[i] = [right, left, modified_op]
                else:
                    assert right in {"1", "PI"}
            else:
                pass
                # raise NotImplementedError(f"all constant for label: {labels[i]}")

    for i, (left, right, op) in enumerate(labels):
        left = left[-1:] if left.startswith("temp_") else left
        right = right[-1:] if right.startswith("temp_") else right
        # if (left == "PI" or right == "PI"):# and op not in {'*', '/'}:
        #     print(labels[i])
        labels[i] = [left, right, op]

    max_temp_org = max([v for v in target_template if v.startswith("temp_")])
    max_temp_update = max([v for v in target_norm_post_template if v.startswith("temp_")])
    gap = ord(max_temp_org[-1]) - ord(max_temp_update[-1])
    if gap > 0:
        for i, (left, right, op) in enumerate(labels):
            left = chr(ord(left) + gap) if len(left) == 1 and ord(left) >= ord('a') and ord(left) <= ord('z') else left
            right = chr(ord(right) + gap) if len(right) == 1 and ord(right) >= ord('a') and ord(right) <= ord('z') else right
            labels[i] = [left, right, op]
    return labels, both_m, gap, got_duplicate

def check_intermediate_m_in_order(labels: List[List[str]]):
    current_m_idx = 0
    for idx, (left_var, right_var, op) in enumerate(labels):
        if left_var.startswith("m_"):
            # try:
            assert int(left_var[2:]) - current_m_idx == 1
            # except:
            #     print("not incremental")
            current_m_idx += 1
    return True


def process_obj(obj: Dict, remove_duplicate: bool = False):
    target_template = [val.strip() for val in obj["target_template"]]

    labels, have_both_m, gap, got_duplicate = get_labels(obj["target_norm_post_template"], obj["target_template"], remove_duplicate)
    type_str = "legal"

    # if count_variable(target_template) > 7: ## only 2 in test
    #     type_str = "variable more than 7"
    #     return type_str, labels, gap


    # if have_constant(target_template):
    #     type_str = "have constant"
    #     return type_str, labels
    #
    # if have_pi(target_template):
    #     type_str = "have pi"
    #     print(obj["equation"], obj["target_template"])
    #     return type_str, labels

    if have_square(target_template): ## only 1 in test
        type_str = "have square"
        return type_str, labels, gap, False

    # if have_both_m:
    #     type_str = "have both m0, m1"
    #     return type_str, labels, gap


    # have_same_variable = []
    # for idx, curr_labels in enumerate(labels):
    #     if curr_labels[0] == curr_labels[1]:
    #         have_same_variable.append(idx)
    # if len(have_same_variable) > 0:
    #     if len(have_same_variable) == 1 and have_same_variable[0] == 0:
    #         return "legal", labels, gap
    #     else:
    #         return "have same variable at multiple layer", labels, gap

    # if have_multiple_m0(target_template):
    #     type_str = "have mutiple m0"
    #     return type_str, labels, gap





    return type_str, labels, gap, got_duplicate

def main():
    remove_duplicate = True
    for in_file in ["train23k_processed.json", "valid23k_processed.json", "test23k_processed.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/math23k/{in_file}"
        if remove_duplicate:
            out_file = in_file.split(".json")[0] + "_nodup_more.json"
        else:
            out_file = in_file.split(".json")[0] + "_all.json"
        data = read_data(in_file)
        count = Counter()
        inst_num_with_gap = 0
        duplicate_num = 0
        for obj in tqdm(data, desc="processing data", total=len(data)):
            type_str, labels, gap, got_duplicate = process_obj(obj, remove_duplicate=remove_duplicate)
            if len(labels) == 0:
                assert len(obj["target_norm_post_template"]) == 3
                print("something", obj["num_list"], obj["equation"])
            if gap > 0:
                inst_num_with_gap += 1
            count[type_str] += 1
            obj["type_str"] = type_str
            obj["equation_layer"] = labels
            duplicate_num += 1 if got_duplicate else 0
            # if type_str == "legal":
            #     check_intermediate_m_in_order(labels)
        # write_data(file=out_file, data = data)

        print(inst_num_with_gap)
        print(f" duplication number: {duplicate_num}")
        for key in count:
            print(f"{key}, valid number: {count[key]}, total: {len(data)}, %: {count[key] * 1.0 / len(data) * 100:.2f}")

def get_five_folds():
    random.seed(42)
    import os
    all_data = []
    for in_file in ["train23k_processed_nodup.json", "valid23k_processed_nodup.json", "test23k_processed_nodup.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/math23k/{in_file}"
        all_data.append(read_data(in_file))
    all_data = all_data[0] + all_data[1] + all_data[2]
    random.shuffle(all_data)
    num_fold = 5
    fold_size = len(all_data) // num_fold
    output_folder= "math23k_five_fold"
    os.makedirs(f"../data/math23k_five_fold", exist_ok=True)
    for i in range(num_fold):
        if i == num_fold - 1:
            test_data = all_data[i * fold_size:]
            train_data = all_data[:i * fold_size]
        else:
            test_data = all_data[i * fold_size:(i + 1) * fold_size]
            train_data = all_data[:i * fold_size] + all_data[(i + 1) * fold_size:]
        size = len(train_data) + len(test_data)
        print(f"total size : {size}, train: {len(train_data)}, test: {len(test_data)}")
        write_data(file=f"../data/{output_folder}/train_{i}.json", data=train_data)
        write_data(file=f"../data/{output_folder}/test_{i}.json", data=test_data)

if __name__ == '__main__':
    # text = "a () * c"
    # print(re.sub(r"\(.*\)", "temp_m", text))
    # main()
    get_five_folds()
    # print(breakit('(((a+b)+a)+c)'))