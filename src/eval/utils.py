


import math
from typing import List

uni_labels = [
    '+','-', '-_rev', '*', '/', '/_rev'
]

def compute(left: float, right:float, op:str):
    if op == "+":
        return left + right
    elif op == "-":
        return left - right
    elif op == "*":
        return left * right
    elif op == "/":
        return (left * 1.0 / right) if right != 0 else  (left * 1.0 / 0.001)
    elif op == "-_rev":
        return right - left
    elif op == "/_rev":
        return (right * 1.0 / left) if left != 0 else  (right * 1.0 / 0.001)
    else:
        raise NotImplementedError(f"not implementad for op: {op}")

def compute_value(equations, num_list, num_constant, constant_values: List[float] = None):
    current_value = 0
    for equation in equations:
        left_var_idx, right_var_idx, op_idx, _ = equation
        left_number = num_list[left_var_idx - num_constant] if left_var_idx >= num_constant else None
        if left_var_idx != -1 and left_var_idx < num_constant: ## means left number is a
            left_number = constant_values[left_var_idx]
        right_number = num_list[right_var_idx - num_constant] if right_var_idx >= num_constant else constant_values[right_var_idx]
        op = uni_labels[op_idx]
        if left_number is None:
            assert current_value is not None
            current_value = compute(current_value, right_number, op)
        else:
            current_value = compute(left_number, right_number, op)
    return current_value

def is_value_correct(predictions, labels, num_list, num_constant, constant_values: List[float] = None):
    pred_val = compute_value(predictions, num_list, num_constant, constant_values)
    gold_val = compute_value(labels, num_list, num_constant, constant_values)
    if math.fabs((gold_val- pred_val)) < 1e-4:
        return True
    else:
        return False