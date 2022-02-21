

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
        if val.strip().startswith("number"):
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

    pointer = 0
    labels = []
    both_m = False
    eq_2_m = {}
    got_duplicate = False
    stack = [target_norm_post_template[0], target_norm_post_template[1], target_norm_post_template[2]]
    pointer = 3
    while len(stack) > 0:
        if len(stack) >= 3 and stack[-1] not in {'+', '-', '*', '/', '^'} and stack[-2] not in {'+', '-', '*', '/', '^'} and stack[-3] in {'+', '-', '*', '/', '^'}:
            if stack[-1].startswith("m_") and stack[-2].startswith("m_"):
                both_m = True
            assert not stack[-3].startswith("number")
            if remove_duplicate:
                checker = check_in_labels([stack[-2], stack[-1], stack[-3]], labels)
                if checker:
                    got_duplicate = True
                    m_string = eq_2_m[' '.join(checker)]
                else:
                    labels.append([stack[-2], stack[-1], stack[-3]])
                    m_string = f"m_{len(labels)}"
                    eq_2_m[' '.join([stack[-2], stack[-1], stack[-3]])] = m_string
            else:
                labels.append([stack[-2], stack[-1], stack[-3]])
                m_string = f"m_{len(labels)}"
            stack.pop()
            stack.pop()
            stack.pop()
            stack.append(m_string)
        else:
            if pointer < len(target_norm_post_template):
                stack.append(target_norm_post_template[pointer])
                pointer += 1
            else:
                assert len(stack) == 1
                stack.pop()
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
                if not left.startswith("number"):
                    labels[i] = [right, str(float(left)), modified_op]
                    const_list.add(str(float(left)))
                    const2num[str(float(left))] +=1
                    contain_constant = True
            else:
                if not right.startswith("number"):
                    labels[i] = [left, str(float(right)), op]
                    const_list.add(str(float(right)))
                    const2num[str(float(right))] += 1
                    contain_constant = True
        else:
            if left.startswith("number") or right.startswith("number"):
                if left.startswith("number") and right.startswith("number"):
                    left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                    modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                    labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
                elif right.startswith("number"):
                    modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                    labels[i] = [right, left, modified_op]
                    const_list.add(str(float(left)))
                    const2num[str(float(left))] += 1
                    contain_constant = True
                else:
                    labels[i] = [left, str(float(right)), op]
                    const_list.add(str(float(right)))
                    const2num[str(float(right))] += 1
                    contain_constant = True
                    #assert right in {"1", "PI"}
            else:
                labels[i] = [str(float(left)), str(float(right)), op]
                const_list.add(str(float(left)))
                const_list.add(str(float(right)))
                const2num[str(float(left))] += 1
                const2num[str(float(right))] += 1
                contain_constant = True
                pass
                # raise NotImplementedError(f"all constant for label: {labels[i]}")

    for i, (left, right, op) in enumerate(labels):
        if left.startswith("number"):
            assert len(left[-1:]) == 1
        if right.startswith("number"):
            assert len(right[-1:]) == 1
        left = chr(ord('a') + int(left[-1:])) if left.startswith("number") else left
        right = chr(ord('a') + int(right[-1:])) if right.startswith("number") else right
        # if (left == "PI" or right == "PI"):# and op not in {'*', '/'}:
        #     print(labels[i])
        labels[i] = [left, right, op]

    temp_var_list = [v for v in target_template if v.startswith("number")]
    gap = 0
    if len(temp_var_list) != 0:
        max_temp_org = max([v for v in target_template if v.startswith("number")])
        max_temp_update = max([v for v in target_norm_post_template if v.startswith("number")])
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
    vars = obj["Equation"].split()
    target_template = [val.strip() for val in vars]

    labels, have_both_m, gap, got_duplicate = get_labels(vars, vars, remove_duplicate)
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

def replace_mawps():
    svamp_train = read_data("../data/mawps_asdiv-a_svamp/trainset.json")
    d1 = read_data("../data/mawps-single/mawps_train_nodup.json")
    d2 = read_data("../data/mawps-single/mawps_valid_nodup.json")
    d3 = read_data("../data/mawps-single/mawps_test_nodup.json")
    mawps_data = d1 + d2 + d3
    # svamp_id2mawps_id = {}
    mapws_id2svamp_id = {}
    mapws_id2obj = {}
    svamp_id2mapws_obj = {}
    num_not_found = 0
    for obj in tqdm(mawps_data, total=len(mawps_data), desc="replace mawps"):
        found = False
        for svamp_obj in svamp_train:
            question = svamp_obj["Question"]
            # if svamp_obj["id"] == "0":
            #     print("sss")
            for i in range(0, 20):
                question = question.replace(f"number{i}", f"temp_{chr(ord('a') + i)}")
            #"Numbers": "25.0 17.0 10.0",
            num_list = [float(val) for val in svamp_obj["Numbers"].split()]
            if question == obj['text'] and num_list == obj["num_list"]:
                found = True
                assert obj["iIndex"] not in mapws_id2svamp_id
                mapws_id2svamp_id[obj["iIndex"]] = svamp_obj["id"]
                mapws_id2obj[obj["iIndex"]] = obj
                # assert svamp_obj["id"] not in svamp_id2mawps_id
                # svamp_id2mawps_id[svamp_obj["id"]] = obj
                break
        try:
            assert found
        except:
            num_not_found +=1
    for mawps_id, svamp_id in mapws_id2svamp_id.items():
        mawps_obj = mapws_id2obj[mawps_id]
        svamp_id2mapws_obj[svamp_id] = mawps_obj
    print(f"num not found : {num_not_found}, num found: {len(mapws_id2svamp_id)} over: {len(mawps_data)}")
    return svamp_id2mapws_obj

def main(use_replace: bool):
    svamp_id2mapws_obj = replace_mawps() if use_replace else {}
    suffix = "_rp" if use_replace else ""
    remove_duplicate = True
    for in_file in ["trainset.json", "testset.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/mawps_asdiv-a_svamp/{in_file}"
        out_file = in_file.split(".json")[0] + f"_nodup{suffix}.json"
        data = read_data(in_file)
        count = Counter()
        inst_num_with_gap = 0
        duplicate_num = 0
        new_data = []
        for obj in tqdm(data, desc="processing data", total=len(data)):
            if obj["id"] in svamp_id2mapws_obj and "train" in in_file:
                new_data.append(svamp_id2mapws_obj[obj["id"]])
                continue
            type_str, labels, gap, got_duplicate = process_obj(obj, remove_duplicate=remove_duplicate)
                # continue
            if len(labels) == 0:
                # assert len(obj["target_norm_post_template"]) == 3
                # print("something", obj["num_list"], obj["equation"])
                print("somehitng")
                pass
            if gap > 0:
                inst_num_with_gap += 1
            count[type_str] += 1
            obj["type_str"] = type_str
            obj["equation_layer"] = labels
            obj["num_list"] = [float(v) for v in obj["Numbers"].split()]
            question = obj["Question"]
            for i in range(10):
                question = question.replace("number"+str(i), "temp_"+chr(ord('a') + i))
            obj["text"] = question
            obj["answer"] = obj.pop("Answer")
            duplicate_num += 1 if got_duplicate else 0
            new_data.append(obj)
            # if type_str == "legal":
            #     check_intermediate_m_in_order(labels)
        write_data(file=out_file, data = new_data)

        print(inst_num_with_gap)
        print(f" duplication number: {duplicate_num}")
        for key in count:
            print(f"{key}, valid number: {count[key]}, total: {len(data)}, %: {count[key] * 1.0 / len(data) * 100:.2f}")
        print(const_list)
        const_list.clear()
        print(sorted(const2num.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
        const2num.clear()

if __name__ == '__main__':
    const_list = set()
    const2num = Counter()
    use_replace = True
    main(use_replace)
    # replace_mawps()