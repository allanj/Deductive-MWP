

import copy

from src.utils import read_data, write_data
from typing import List, Dict
import re
from tqdm import tqdm
from collections import Counter

def check_stat(data: List):
    ops = set()
    digit_list = {chr(i) for i in range(ord('0'), ord('9') + 1)}
    auxs = {'(', ')'}
    for obj in tqdm(data, desc="parsing equation", total=len(data)):
        equation = obj["lEquations"][0]
        assert len(obj["lEquations"]) == 1
        equation = equation.replace("X=", "").replace("x=", "").replace("=x", "").replace(".0", "")
        for idx, c in enumerate(equation):
            if c not in digit_list and c not in auxs:
                # if c == ".":
                #     print(obj)
                if c== "x":
                    print(obj)
                ops.add(c)
    print(ops)

def find_number(data: List):
    pattern = r"\d+\.?\d*"
    num_obj_with_num_not_found = 0
    for obj in tqdm(data, desc="parsing equation", total=len(data)):
        equation = obj["lEquations"][0]
        question = obj["sQuestion"]
        all_numbers = re.findall(pattern=pattern, string=equation)
        num_not_found = False
        for number in all_numbers:
            pure_number_not_found = number not in question
            simplified_number =  number
            if int(float(number)) == float(number):
                simplified_number = str(int(float(number)))
            simplified_number_not_found = simplified_number not in question
            if pure_number_not_found and simplified_number_not_found:
                num_not_found= True
        if num_not_found:
            print(obj)
            num_obj_with_num_not_found += 1
    print(f"number ojects with num not found: {num_obj_with_num_not_found}")

def split_processed_file(processed_file: str, orig_train: str, orig_valid:str, orig_test: str):
    orig_train_data = read_data(orig_train)
    orig_valid_data = read_data(orig_valid)
    orig_test_data = read_data(orig_test)
    train_ids = {obj["iIndex"] for obj in orig_train_data}
    valid_ids = {obj["iIndex"] for obj in orig_valid_data}
    test_ids = {obj["iIndex"] for obj in orig_test_data}
    assert len(train_ids) == len(orig_train_data)
    assert len(valid_ids) == len(orig_valid_data)
    assert len(test_ids) == len(orig_test_data)
    updated_trains = []
    updated_valids = []
    updated_tests = []
    not_exist_num= 0
    for obj in read_data(processed_file):
        # try:
        #     assert obj["iIndex"] in train_ids or  obj["iIndex"] in test_ids or  obj["iIndex"] in valid_ids
        # except:
        #     print(obj["iIndex"])
        #     not_exist_num +=1
        #     continue
        if obj["iIndex"] in train_ids:
            updated_trains.append(obj)
        elif obj["iIndex"] in valid_ids:
            updated_valids.append(obj)
        elif obj["iIndex"] in test_ids:
            updated_tests.append(obj)
        else:
            continue
    assert len(updated_trains) == len(orig_train_data)
    assert len(updated_valids) == len(orig_valid_data)
    assert len(updated_tests) == len(orig_test_data)
    print(f"train: {len(updated_trains)}, dev: {len(updated_valids)}, test: {len(updated_tests)}")
    write_data(file="../data/mawps-single/mawps_train.json", data= updated_trains)
    write_data(file="../data/mawps-single/mawps_valid.json", data=updated_valids)
    write_data(file="../data/mawps-single/mawps_test.json", data=updated_tests)

if __name__ == '__main__':
    split_processed_file(processed_file="../data/mawps-single/new_ma.json",
                         orig_train="../data/mawps-single/trainset.json",
                         orig_valid="../data/mawps-single/validset.json",
                         orig_test="../data/mawps-single/testset.json")
    # # train_data = read_data("../data/mawps-single/trainset.json")
    # valid_data = read_data("../data/mawps-single/validset.json")
    # test_data = read_data("../data/mawps-single/testset.json")
    # # find_number(train_data)
    # find_number(valid_data)
    # find_number(test_data)
    # #
    # check_stat(train_data + valid_data + test_data)
    # pattern = r"\d+\.?\d*"
    # text = "X=(12.0/6.0)"
    # print(re.findall(pattern=pattern, string=text))
