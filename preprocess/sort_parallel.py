import copy

from src.utils import read_data, write_data
from typing import List, Dict
import re
from tqdm import tqdm
from collections import Counter
from functools import cmp_to_key

uni_labels = [
    '+','-', '-_rev', '*', '/', '/_rev'
]

def get_min_if_both_m(x, y):
    height_idx_1, sub_idx_1 = x.split("_")
    height_idx_2, sub_idx_2 = y.split("_")
    if height_idx_1 > height_idx_2:
        eq_a_min = x
    elif height_idx_1 < height_idx_2:
        eq_a_min = y
    else:
        if sub_idx_1 < sub_idx_2:
            eq_a_min = x
        else:
            eq_a_min = y
    return eq_a_min

def get_max_if_both_m(x, y):
    height_idx_1, sub_idx_1 = x.split("_")
    height_idx_2, sub_idx_2 = y.split("_")
    if height_idx_1 < height_idx_2:
        eq_max = x
    elif height_idx_1 > height_idx_2:
        eq_max = y
    else:
        if sub_idx_1 > sub_idx_2:
            eq_max = x
        else:
            eq_max = y
    return eq_max

def compare_eq(eq_a:List[str],eq_b:List[str]):
    updated_eq_a = [x.replace("m_", "") for x in eq_a]
    updated_eq_b = [x.replace("m_", "") for x in eq_b]
    updated_a_label = uni_labels.index(eq_a[2])
    # if updated_eq_a[0] > updated_eq_a[1] and eq_a[2] not in {"+", "*"}:
    #     updated_a_label = uni_labels.index(eq_a[2] + "_rev")
    try:
        if not ("_" in updated_eq_a[0] and "_" in updated_eq_a[1]):
            assert updated_eq_a[0] <= updated_eq_a[1]
        else:
            left_big, left_small = updated_eq_a[0].split("_")
            right_big, right_small = updated_eq_a[1].split("_")
            try:
                assert left_big > right_big or (left_big==right_big and left_small <= right_small)
            except:
                print("inside")
    except:
        print("sss")
    updated_b_label = uni_labels.index(eq_b[2])
    try:
        if not ("_" in updated_eq_b[0] and "_" in updated_eq_b[1]):
            assert updated_eq_b[0] <= updated_eq_b[1]
        else:
            left_big, left_small = updated_eq_b[0].split("_")
            right_big, right_small = updated_eq_b[1].split("_")
            try:
                assert left_big > right_big or (left_big == right_big and left_small <= right_small)
            except:
                print("inside")
    except:
        print("sss")
    # if updated_eq_b[0] > updated_eq_b[1] and eq_b[2] not in {"+", "*"}:
    #     updated_b_label = uni_labels.index(eq_b[2] + "_rev")

    eq_a_min = min(updated_eq_a[0], updated_eq_a[1])
    if "_" in updated_eq_a[0] and "_" in updated_eq_a[1]:
        eq_a_min = get_min_if_both_m(updated_eq_a[0], updated_eq_a[1])
    eq_b_min = min(updated_eq_b[0], updated_eq_b[1])
    if "_" in updated_eq_b[0] and "_" in updated_eq_b[1]:
        eq_b_min = get_min_if_both_m(updated_eq_b[0], updated_eq_b[1])
    if eq_a_min < eq_b_min:
        return -1
    elif eq_a_min > eq_b_min:
        return 1
    else:
        ## min equal.
        eq_a_max = max(updated_eq_a[0], updated_eq_a[1])
        if "_" in updated_eq_a[0] and "_" in updated_eq_a[1]:
            eq_a_max = get_max_if_both_m(updated_eq_a[0], updated_eq_a[1])
        eq_b_max = max(updated_eq_b[0], updated_eq_b[1])
        if "_" in updated_eq_b[0] and "_" in updated_eq_b[1]:
            eq_b_max = get_max_if_both_m(updated_eq_b[0], updated_eq_b[1])
        if eq_a_max < eq_b_max:
            return -1
        elif eq_a_max > eq_b_max:
            return 1
        else:
            ## a_min = b_min, a_max = b_max
            if updated_a_label < updated_b_label:
                return -1
            elif updated_a_label > updated_b_label:
                return 1
            else:
                raise NotImplementedError(f"the equation is exactly the same? {eq_a}, {eq_b}")

def sort_obj(obj: Dict):
    # if obj["id"] == '163':
    #     print("ss")
    assert all(["PI" < chr(x) for x in range(ord('a'), ord('z') + 1) ])
    assert all(["1" < chr(x) for x in range(ord('a'), ord('z') + 1) ])
    assert all([f"{i}_{j}" < chr(x) for i in range(0, 20) for j in range(0,20) for x in range(ord('a'), ord('z') + 1)])
    assert all([f"{i}_{j}" < chr(x) for i in range(0, 20) for j in range(0, 20) for x in range(ord('@'), ord('@') + 1)])
    all([f"{i}_{j}" < x for i in range(0, 20) for j in range(0, 20) for x in ['PI']])
    assert  "@" < "PI"
    parallel_equations = obj["equation_layer"]
    candidate_equations = copy.deepcopy(parallel_equations)
    var_and_const_set = {'@', "PI"}
    if obj["id"] == "3978":
        print("ss")
    for var_id in range(ord('a'), ord('z')+1):
        var_and_const_set.add(chr(var_id))
    for h_idx, _ in enumerate(candidate_equations):
        equations = candidate_equations[h_idx]
        have_special_char = False
        for sub_eq in equations:
            if "^" in sub_eq:
                have_special_char = True
                break
        if have_special_char:
            continue
        for sub_eidx, sub_eq in enumerate(equations):
            if sub_eq[0] == '1':
                sub_eq[0] = '@'
            if sub_eq[1] == '1':
                sub_eq[1] = '@'
            if sub_eq[0] in var_and_const_set and sub_eq[1] in var_and_const_set:
                if sub_eq[0] > sub_eq[1]:
                    op = sub_eq[2] if sub_eq[2] in {'+', '*'} else sub_eq[2] + "_rev"
                    equations[sub_eidx] = [sub_eq[1], sub_eq[0], op]
            elif sub_eq[0].startswith("m_") or sub_eq[1].startswith("m_"):
                if sub_eq[0].startswith("m_") and sub_eq[1].startswith("m_"):
                    left_big, left_small = sub_eq[0].split("_")[-2:]
                    right_big, right_small = sub_eq[1].split("_")[-2:]
                    if left_big > right_big:
                        pass
                    elif left_big < right_big:
                        op = sub_eq[2] if sub_eq[2] in {'+', '*'} else sub_eq[2] + "_rev"
                        equations[sub_eidx] = [sub_eq[1], sub_eq[0], op]
                    else:
                        if left_small > right_small:
                            op = sub_eq[2] if sub_eq[2] in {'+', '*'} else sub_eq[2] + "_rev"
                            equations[sub_eidx] = [sub_eq[1], sub_eq[0], op]
                else:
                    if sub_eq[1].startswith("m_"):
                        op = sub_eq[2] if sub_eq[2] in {'+', '*'} else sub_eq[2] + "_rev"
                        equations[sub_eidx] = [sub_eq[1], sub_eq[0], op]

        ## update current equations
        sorted_equations = sorted(equations,key=cmp_to_key(compare_eq))
        candidate_equations[h_idx] = sorted_equations

        org_idx2new_idx = []
        for org_idx, unsorted_eq in enumerate(equations):
            assert len(unsorted_eq) == 3
            for new_idx, sorted_eq in enumerate(sorted_equations):
                assert len(sorted_eq) == 3
                if unsorted_eq == sorted_eq:
                    org_idx2new_idx.append(new_idx)
        assert len(org_idx2new_idx) == len(equations)

        ## according to the current update, update all after equations
        for c_h_idx in range(h_idx + 1, len(candidate_equations)):
            for e_idx, equation in enumerate(candidate_equations[c_h_idx]):
                for val_idx, element in enumerate(equation):
                    if element.startswith(f"m_{h_idx}_"):
                        org_eq_idx = int(element.split("_")[-1])
                        equation[val_idx] = f"m_{h_idx}_{org_idx2new_idx[org_eq_idx]}"
    for h_idx, _ in enumerate(candidate_equations):
        equations = candidate_equations[h_idx]
        for sub_eidx, sub_eq in enumerate(equations):
            if sub_eq[0] == '@':
                sub_eq[0] = '1'
            if sub_eq[1] == '@':
                sub_eq[1] = '1'
    return candidate_equations


def main():
    demo_file = ["train23k_parallel.json", "valid23k_parallel.json", "test23k_parallel.json"]
    # demo_file = [ "test23k_parallel.json", "valid23k_parallel.json"]
    for in_file in demo_file:
        print(f"working on... {in_file}")
        in_file = f"../data/math23k/{in_file}"
        out_file = in_file.split(".json")[0] + "_sorted.json"
        data = read_data(in_file)
        count = Counter()
        for obj in tqdm(data, desc="processing data", total=len(data)):
            sorted_eqs = sort_obj(obj)
            obj["sorted_equation_layer"] = sorted_eqs
        write_data(file=out_file, data = data)

        for key in count:
            print(f"{key}, valid number: {count[key]}, total: {len(data)}, %: {count[key] * 1.0 / len(data) * 100:.2f}")


def test():
    x =     [

                                "a",

                                "m_0_0",

                                "+"

                            ]
    y = ["m_0_0", "a", "-"]
    print(compare_eq(x,y))

if __name__ == '__main__':
    main()
    # test()