from src.utils import read_data, write_data
from typing import List, Dict
import re
from tqdm import tqdm
from collections import Counter

from preprocess.process_math23k import have_square

def process_obj(obj: Dict):
    target_template = [val.strip() for val in obj["target_template"]]
    type_str = "legal"
    if have_square(target_template): ## only 1 in test
        type_str = "have square"
    if obj["parallel_equation_layer"] is None or len(obj["parallel_equation_layer"]) == 0:
        type_str = "empty equation"
    equation_layers = []
    for equations in obj["parallel_equation_layer"]:
        curr_eqs = []
        for sub_equation in equations:
            element = sub_equation.split(" ")
            left, op, right = element
            if "_" in left:
                left = "m_"  + left
            if "_" in right:
                right = "m_" + right
            curr_eqs.append([left, right, op])
        equation_layers.append(curr_eqs)
    obj["equation_layer"] = equation_layers
    return type_str


def main():
    for in_file in ["train23k_processed_parallel.json", "valid23k_processed_parallel.json", "test23k_processed_parallel.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/math23k/{in_file}"
        out_file = in_file.split(".json")[0].replace("_processed", "") + ".json"
        data = read_data(in_file)
        count = Counter()
        for obj in tqdm(data, desc="processing data", total=len(data)):
            type_str = process_obj(obj)
            count[type_str] += 1
            obj["type_str"] = type_str
        write_data(file=out_file, data = data)

        for key in count:
            print(f"{key}, valid number: {count[key]}, total: {len(data)}, %: {count[key] * 1.0 / len(data) * 100:.2f}")

if __name__ == '__main__':
    main()