


import math
from typing import List

# uni_labels = [
#     '+','-', '-_rev', '*', '/', '/_rev'
# ]

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
    elif op == "^":
        try:
            return math.pow(left, right)
        except:
            return 0
    elif op == "^_rev":
        try:
            return math.pow(right, left)
        except:
            return 0
    else:
        raise NotImplementedError(f"not implementad for op: {op}")

def compute_value(equations, num_list, num_constant, uni_labels, constant_values: List[float] = None):
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


def compute_value_for_incremental_equations(equations, num_list, num_constant, uni_labels, constant_values: List[float] = None):
    current_value = 0
    store_values = []
    grounded_equations = []
    for eq_idx, equation in enumerate(equations):
        left_var_idx, right_var_idx, op_idx, _ = equation
        assert left_var_idx >= 0
        assert right_var_idx >= 0
        if left_var_idx >= eq_idx and left_var_idx < eq_idx + num_constant:  ## means
            left_number = constant_values[left_var_idx - eq_idx]
        elif left_var_idx >= eq_idx + num_constant:
            left_number = num_list[left_var_idx - num_constant - eq_idx]
        else:
            assert left_var_idx < eq_idx  ## means m
            m_idx = eq_idx - left_var_idx
            left_number = store_values[m_idx - 1]

        if right_var_idx >= eq_idx and right_var_idx < eq_idx + num_constant:## means
            right_number = constant_values[right_var_idx- eq_idx]
        elif right_var_idx >= eq_idx + num_constant:
            right_number = num_list[right_var_idx - num_constant - eq_idx]
        else:
            assert right_var_idx < eq_idx ## means m
            m_idx = eq_idx - right_var_idx
            right_number = store_values[m_idx - 1]

        op = uni_labels[op_idx]
        current_value = compute(left_number, right_number, op)
        grounded_equations.append([left_number, right_number, op, current_value])
        store_values.append(current_value)
    return current_value, grounded_equations

def compute_value_for_parallel_equations(parallel_equations:List[List], num_list, num_constant, uni_labels, constant_values: List[float] = None):
    current_value = 0
    store_values = []
    grounded_equations = []
    accumulate_eqs = [0]
    for p_idx, equations in enumerate(parallel_equations):
        current_store_values = []
        for eq_idx, equation in enumerate(equations):
            left_var_idx, right_var_idx, op_idx, _ = equation
            assert left_var_idx >= 0
            assert right_var_idx >= 0
            if left_var_idx >= accumulate_eqs[p_idx] and left_var_idx < accumulate_eqs[p_idx] + num_constant:  ## means
                left_number = constant_values[left_var_idx - accumulate_eqs[p_idx]]
            elif left_var_idx >= accumulate_eqs[p_idx] + num_constant:
                left_number = num_list[left_var_idx - num_constant - accumulate_eqs[p_idx]]
            else:
                assert left_var_idx < accumulate_eqs[p_idx]  ## means m
                m_idx = accumulate_eqs[p_idx] - left_var_idx
                left_number = store_values[left_var_idx]

            if right_var_idx >= accumulate_eqs[p_idx] and right_var_idx < accumulate_eqs[p_idx] + num_constant:## means
                right_number = constant_values[right_var_idx- accumulate_eqs[p_idx]]
            elif right_var_idx >= accumulate_eqs[p_idx] + num_constant:
                right_number = num_list[right_var_idx - num_constant - accumulate_eqs[p_idx]]
            else:
                assert right_var_idx < accumulate_eqs[p_idx] ## means m
                m_idx = accumulate_eqs[p_idx] - right_var_idx
                right_number = store_values[right_var_idx]

            op = uni_labels[op_idx]
            current_value = compute(left_number, right_number, op)
            grounded_equations.append([left_number, right_number, op, current_value])
            current_store_values.append(current_value)
        store_values = current_store_values + store_values
        accumulate_eqs.append(accumulate_eqs[len(accumulate_eqs) - 1] + len(equations))
    return current_value, grounded_equations

def is_value_correct(predictions, labels, num_list, num_constant, uni_labels, constant_values: List[float] = None, consider_multiple_m0=False, use_parallel_equations: bool = False):
    pred_grounded_equations = None
    gold_grounded_equations = None
    if consider_multiple_m0:
        if use_parallel_equations:
            pred_val, pred_grounded_equations = compute_value_for_parallel_equations(predictions, num_list, num_constant, uni_labels, constant_values)
        else:
            pred_val, pred_grounded_equations = compute_value_for_incremental_equations(predictions, num_list, num_constant, uni_labels, constant_values)
    else:
        pred_val = compute_value(predictions, num_list, num_constant, uni_labels, constant_values)
    if consider_multiple_m0:
        if use_parallel_equations:
            gold_val, gold_grounded_equations = compute_value_for_parallel_equations(labels, num_list, num_constant, uni_labels, constant_values)
        else:
            gold_val, gold_grounded_equations = compute_value_for_incremental_equations(labels, num_list, num_constant, uni_labels,  constant_values)
    else:
        gold_val = compute_value(labels, num_list, num_constant, uni_labels, constant_values)
    if math.fabs((gold_val- pred_val)) < 1e-4:
        return True, pred_val, gold_val, pred_grounded_equations, gold_grounded_equations
    else:
        return False, pred_val, gold_val, pred_grounded_equations, gold_grounded_equations