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
    # assert target_norm_post_template[:2] == ["X", "="] or target_norm_post_template[:2] == ["x", "="]
    # if len(target_norm_post_template) == 3:
    #     assert target_norm_post_template[2].startswith("temp_")
    #     target_norm_post_template.append("1")
    #     target_norm_post_template.append("*")
    stack = []
    pointer = 0
    labels = []
    both_m = False
    eq_2_m = {}
    contain_constant = False
    while pointer != len(target_norm_post_template):
        stack.append(target_norm_post_template[pointer])
        if stack[-1] in {'+', '-', '*', '/', '^'}:
            if len(stack[-3:]) == 3:
                if stack[-3].startswith("m_") and stack[-2].startswith("m_"):
                    both_m = True
                if remove_duplicate:
                    checker = check_in_labels([stack[-3], stack[-2], stack[-1]], labels)
                    if checker:
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
        left = "3.14" if left == "PI" else left
        right = "3.14" if right == "PI" else right
        if left.startswith("m_") or right.startswith("m_"):
            if left.startswith("m_") and right.startswith("m_"):
                left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
            elif right.startswith("m_"):
                modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                labels[i] = [right, left, modified_op]
                if not left.startswith("temp_"):
                    labels[i] = [right, str(float(left)), modified_op]
                    const_list.add(str(float(left)))
                    const2num[str(float(left))] +=1
                    contain_constant = True
            else:
                if not right.startswith("temp_"):
                    labels[i] = [left, str(float(right)), op]
                    const_list.add(str(float(right)))
                    const2num[str(float(right))] += 1
                    contain_constant = True
        else:
            if left.startswith("temp_") or right.startswith("temp_"):
                if left.startswith("temp_") and right.startswith("temp_"):
                    left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                    modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                    labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
                elif right.startswith("temp_"):
                    modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                    labels[i] = [right, str(float(left)), modified_op]
                    const_list.add(str(float(left)))
                    const2num[str(float(left))] += 1
                    contain_constant = True
                else:
                    # pass
                    labels[i] = [left, str(float(right)), op]
                    const_list.add( str(float(right)))
                    const2num[ str(float(right))] += 1
                    contain_constant = True
                    # assert right in {"1", "PI", "12"}
            else:
                labels[i] = [str(float(left)), str(float(right)), op]
                const_list.add(str(float(left)))
                const_list.add(str(float(right)))
                const2num[str(float(left))] += 1
                const2num[str(float(right))] += 1
                contain_constant = True
                # print("be "labels[i]) ## both are constant
                pass
                # raise NotImplementedError(f"all constant for label: {labels[i]}")

    for i, (left, right, op) in enumerate(labels):
        left = left[-1:] if left.startswith("temp_") else left
        right = right[-1:] if right.startswith("temp_") else right
        # if (left == "PI" or right == "PI"):# and op not in {'*', '/'}:
        #     print(labels[i])
        labels[i] = [left, right, op]

    temp_var_list = [v for v in target_template if v.startswith("temp_")]
    gap = 0
    if len(temp_var_list) !=0:
        max_temp_org = max(temp_var_list)
        max_temp_update = max([v for v in target_norm_post_template if v.startswith("temp_")])
        gap = ord(max_temp_org[-1]) - ord(max_temp_update[-1])
        if gap > 0:
            for i, (left, right, op) in enumerate(labels):
                left = chr(ord(left) + gap) if len(left) == 1 and ord(left) >= ord('a') and ord(left) <= ord('z') else left
                right = chr(ord(right) + gap) if len(right) == 1 and ord(right) >= ord('a') and ord(right) <= ord('z') else right
                labels[i] = [left, right, op]
    return labels, both_m, gap, contain_constant

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
    target_template = [val.strip() for val in obj["mapped_equation"].split()]

    try:
        labels, have_both_m, gap, contain_constant = get_labels(obj["posted_equation"].split(), obj["mapped_equation"].split(), remove_duplicate)
        type_str = "legal"
    except:
        print(obj['id'])
        type_str = "error getting labels"
        return type_str, [], 0, False

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





    return type_str, labels, gap, contain_constant

def main():
    remove_duplicate = True
    for in_file in ["mwp_processed.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/large_math/{in_file}"
        out_file = in_file.split(".json")[0] + "_extracted.json"
        data = read_data(in_file)
        count = Counter()
        inst_num_with_gap = 0
        num_cannot_compute_labels = 0
        num_inst_have_constants = 0
        for obj in tqdm(data, desc="processing data", total=len(data)):
            type_str, labels, gap, contain_constant = process_obj(obj, remove_duplicate=remove_duplicate)
            if len(labels) == 0:
                # print(obj)
                num_cannot_compute_labels+=1
            obj["have_constant"] = contain_constant
            if contain_constant:
                # print(obj)
                num_inst_have_constants+=1
                # assert len(obj["norm_post_equ"]) == 3
                # print("something", obj["num_list"], obj["norm_mid_equ"])
            if gap > 0:
                inst_num_with_gap += 1
            count[type_str] += 1
            obj["type_str"] = type_str
            obj["equation_layer"] = labels
            obj["answer"] = obj["ans"]
            obj.pop("ans")
            obj["text"] = obj["mapped_text"]
            obj.pop("mapped_text")
            elements = obj["text"].split()
            new_elements = []
            for idx, element in enumerate(elements):
                if idx > 0 and elements[idx - 1] == "@":
                    continue
                if element == "@":
                    assert ord(elements[idx+1]) >= ord('a') and ord(elements[idx+1]) <= ord('z')
                    new_elements.append(f"temp_{elements[idx+1]}")
                else:
                    new_elements.append(element)
            obj["text"] = ' '.join(new_elements)
            # if type_str == "legal":
            #     check_intermediate_m_in_order(labels)
        write_data(file=out_file, data = data)

        print(inst_num_with_gap)
        for key in count:
            print(f"{key}, valid number: {count[key]}, total: {len(data)}, %: {count[key] * 1.0 / len(data) * 100:.2f}")
        print(f"number cannot compute: {num_cannot_compute_labels}, num insts have constant: {num_inst_have_constants}")
        print(const_list)
        const_list.clear()
        print(sorted(const2num.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        const2num.clear()

def split_the_filtered_files():
    in_file = f"../data/large_math/mwp_processed_filtered.json"
    data = read_data(in_file)
    random.seed(42)
    total = len(data)
    random.shuffle(data)
    train_num = int(total * 0.8)
    valid_num = int(total * 0.1)
    train_data = data[:train_num]
    valid_data = data[train_num:(train_num+valid_num)]
    test_data = data[(train_num+valid_num):]
    write_data(file = f"../data/large_math/large_math_train_nodup.json", data=train_data)
    write_data(file=f"../data/large_math/large_math_valid_nodup.json", data=valid_data)
    write_data(file=f"../data/large_math/large_math_test_nodup.json", data=test_data)


if __name__ == '__main__':
    const_list = set()
    const2num = Counter()
    main()
    print(const_list)

    # split_the_filtered_files()