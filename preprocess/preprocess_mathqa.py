
from src.utils import read_data, write_data
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
from preprocess.mathqa_utils import parse_number, label2op,func2op, constants, parse_answer
from typing import List, Tuple
import re
# \u03c0  -> pi

filtered_ops = {
'floor', 'choose', 'min', 'tangent', 'sine', 'reminder', 'lcm', 'factorial',
 'gcd', 'max', 'permutation', 'triangle_area_three_edges', 'surface_cylinder', 'rhombus_perimeter',
 'surface_rectangular_prism', 'speed_in_still_water', 'log', 'negate',
    ## following are not exists
    'round', 'cosine', 'circle_arc', 'degree_to_radians', 'radians_to_degree', 'sum_consecutive_number',
    'semi_circle_perimiter', 'circle_sector_area'## not exists
}

def replace_question(problem_string:str ,  spans: List[Tuple]):
    new_problem = []
    current_num_idx = 0
    current_span = spans[current_num_idx]
    char_idx = 0
    while char_idx < len(problem_string):
        if current_span is not None and char_idx == current_span[0]:
            new_problem.append(f" temp_{chr(ord('a') + current_num_idx)} ")
            char_idx += current_span[1] - current_span[0]
            current_num_idx += 1
            current_span = spans[current_num_idx] if current_num_idx < len(spans) else None
            continue
        else:
            new_problem.append(problem_string[char_idx])
        char_idx += 1
    new_problem = ''.join(new_problem)
    return new_problem

def get_var(arg_var):
    if arg_var.startswith("const_"):
        return constants[arg_var]
    elif arg_var.startswith("n"):
        num_idx = int(arg_var[1:])
        return f"temp_{chr(ord('a') + num_idx)}"
    elif arg_var.startswith("#"):
        num_idx = int(arg_var[1:])
        return f"globalm_{num_idx}"
    else:
        raise NotImplementedError(f"not implemented for variable: {arg_var}")

def get_processed_equation(equation: str, intermediate_offset:int, current_equation_idx: int):
    eles = equation.split(",")
    op_name = eles[0].split("(")[0]
    args_a = eles[0].split("(")[1]
    args = [args_a]
    eq_num = 0
    if len(eles) > 1:
        if len(eles) == 2:
            assert eles[1].endswith(")")
            args.append(eles[1][:-1])
        elif len(eles) == 3:
            assert eles[2].endswith(")")
            args.append(eles[1])
            args.append(eles[2][:-1])
    else:
        args[0] = args[0][:-1] ## to remove the final )
    modified_op_name = func2op[op_name]
    if isinstance(modified_op_name, str):
        operator = label2op[modified_op_name]
        label = [[get_var(args[0]), get_var(args[1]), operator]]
        eq_num = 1
    elif isinstance(modified_op_name, Tuple):
        operator = label2op[modified_op_name[0]]
        vars = []
        for name in [modified_op_name[1], modified_op_name[2]]:
            if isinstance(name, str) and name.startswith("temp_"):
                var_idx = ord(name.replace("temp_", "")) - ord('a')
                current_arg = args[var_idx]
                current_var = get_var(current_arg)
            else:
                current_var = name if name != 'const_pi' else constants[name]
                assert isinstance(current_var, float) or isinstance(current_var, int)
            vars.append(current_var)
        label = [vars + [operator]]
        eq_num = 1
    elif isinstance(modified_op_name, list):
        label = []
        for t_idx, eq_tuple in enumerate(modified_op_name):
            operator = label2op[eq_tuple[0]]
            vars = []
            for name in [eq_tuple[1], eq_tuple[2]]:
                if isinstance(name, str) and name.startswith("temp_"):
                    var_idx = ord(name.replace("temp_", "")) - ord('a')
                    current_arg = args[var_idx]
                    current_var = get_var(current_arg)
                elif isinstance(name, str) and name.startswith("m_"):
                    current_var = name.replace("m_", "localm_")
                    # current_intermediate_idx = int(name.replace("m_", "localm_"))
                    # current_var =  f"m_{current_equation_idx + current_intermediate_idx}"
                else:
                    current_var = name if name != 'const_pi' else constants[name]
                    assert isinstance(current_var, float) or isinstance(current_var, int)
                vars.append(current_var)
            current_label = vars + [operator]
            label.append(current_label)
        # intermediate_offset += len(modified_op_name)
        eq_num = len(modified_op_name)
    else:
        raise NotImplementedError(f"not implemented for type: {modified_op_name}")

    return label, intermediate_offset, eq_num

def check_maximum_num_list(equations: List[str]):
    max_idx = -1
    for eq_idx, equation in enumerate(equations):
        assert equation.endswith(")")
        start = equation.index("(")
        var_str = equation[start+1:-1]
        var_list = var_str.split(",")
        for var in var_list:
            if var.startswith("n"):
                max_idx = max(int(var.replace("n", "")), max_idx)
    return max_idx

def convert_equations(equations: List[str]):
    all_equation_layers = []
    for eq_idx, equation in enumerate(equations):
        sub_equations, intermediate_offset, sub_eq_num = get_processed_equation(equation=equation, intermediate_offset=0, current_equation_idx=eq_idx)
        all_equation_layers.append(sub_equations)
    ## post process, first, update_global_m first
    accumulate_prev_extra  = [0]
    accumulate_prev_all = [0]
    for global_idx, global_equations in enumerate(all_equation_layers):
        accumulate_prev_extra.append(len(global_equations) - 1 + accumulate_prev_extra[global_idx])
        accumulate_prev_all.append(len(global_equations) + accumulate_prev_all[global_idx])
    accumulate_prev_extra = accumulate_prev_extra[1:]
    accumulate_prev_all = accumulate_prev_all[:-1]
    for global_idx, global_equations in enumerate(all_equation_layers):
        for local_idx, sub_equation in enumerate(global_equations):
            for idx, var in enumerate([sub_equation[0], sub_equation[1]]):
                if isinstance(var, str) and var.startswith("globalm_"):
                    sub_equation[idx] = f"globalm_{accumulate_prev_extra[int(var.replace('globalm_', ''))] + int(var.replace('globalm_', ''))}"
    ## now update the localm
    for global_idx, global_equations in enumerate(all_equation_layers):
        for local_idx, sub_equation in enumerate(global_equations):
            for idx, var in enumerate([sub_equation[0], sub_equation[1]]):
                if isinstance(var, str) and var.startswith("localm_"):
                    sub_equation[idx] = f"globalm_{accumulate_prev_all[global_idx] + int(var.replace('localm_', ''))}"
    all_equation_layers = [equation for equation_layer in all_equation_layers for equation in equation_layer]
    for global_idx, global_eq in enumerate(all_equation_layers):
        for idx, var in enumerate([global_eq[0], global_eq[1]]):
            if isinstance(var, str) and var.startswith("globalm_"):
                global_eq[idx] = f"m_{int(var.replace('globalm_', ''))}"
    return all_equation_layers


def process_options_and_answers(option_string, answer_string):
    options = [item for item in re.findall('[a-e] \) ([^,]*)', option_string)]
    current_answer_string_in_option = options[ord(answer_string) - ord('a')]
    answer= parse_answer(current_answer_string_in_option)
    if answer == "none":
        # print(option_string, answer)
        return None
    if answer.endswith(" / 00"):
        answer = answer.replace(" / 00", "").strip()
    try:
        return eval(answer)
    except:
        # print(option_string, answer)
        return None
    # print(answer, eval(answer))


def process_all_question(all_equation_layers):
    labels = all_equation_layers
    ## temp_ m_ "const_" -> constant value
    for i, (left, right, op) in enumerate(all_equation_layers):
        left, right = str(left), str(right)
        if left.startswith("m_") or right.startswith("m_"):
            if left.startswith("m_") and right.startswith("m_"):
                left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                modified_op = op + "_rev" if op in {'-', '/', '**'} and (not left_is_smaller) else op
                labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
            elif right.startswith("m_"):
                modified_op = op + "_rev" if op in {'-', '/', '**'} else op
                labels[i] = [right, left, modified_op]
                if not left.startswith("temp_"):
                    labels[i] = [right, str(float(left)), modified_op]
                    const_list.add(str(float(left)))
                    const2num[str(float(left))] += 1
            else:
                if not right.startswith("temp_"):
                    labels[i] = [left, str(float(right)), op]
                    const_list.add(str(float(right)))
                    const2num[str(float(right))] += 1
        else:
            if left.startswith("temp_") or right.startswith("temp_"):
                if left.startswith("temp_") and right.startswith("temp_"):
                    left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                    modified_op = op + "_rev" if op in {'-', '/', '**'} and (not left_is_smaller) else op
                    labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
                elif right.startswith("temp_"):
                    modified_op = op + "_rev" if op in {'-', '/', '**'} else op
                    labels[i] = [right, str(float(left)), modified_op]
                    const_list.add(str(float(left)))
                    const2num[str(float(left))] += 1
                else:
                    labels[i] = [left, str(float(right)), op]
                    const_list.add(str(float(right)))
                    const2num[str(float(right))] += 1
            else:
                labels[i] = [str(float(left)), str(float(right)), op]
                const_list.add(str(float(left)))
                const_list.add(str(float(right)))
                const2num[str(float(left))] += 1
                const2num[str(float(right))] += 1
                pass

    for i, (left, right, op) in enumerate(labels):
        left = left[-1:] if left.startswith("temp_") else left
        right = right[-1:] if right.startswith("temp_") else right
        if left.startswith("m_"):
            left = f"m_{int(left.replace('m_', '')) + 1}"
        if right.startswith("m_"):
            right = f"m_{int(right.replace('m_', '')) + 1}"
        if op.startswith("**"):
            op = op.replace("**", "^")
        labels[i] = [left, right, op]

def process_obj(obj):
    linear_formula = obj["linear_formula"]
    problem_string = obj["Problem"]
    equations = linear_formula.split("|")
    equations = [eq for eq in equations if eq.strip() != '']
    num_list, spans = parse_number(problem_string)
    new_problem = replace_question(problem_string, spans) if len(spans) > 0 else problem_string
    obj["text"] = new_problem
    obj["num_list"] = num_list
    total_nums = check_maximum_num_list(equations) + 1
    assert len(num_list) >= total_nums

    type_str = "legal"
    obj["type_str"] = "legal"
    for equation in equations:
        eles = equation.split(",")
        # if len(eles) != 1 and len(eles) != 2:
        #     type_str = "illegal: more than 2 var"
        #     # print(obj["id"])
        #     break
        op_name = eles[0].split("(")[0]
        if op_name in filtered_ops:
            type_str = "illegal_operator"
            # print(obj["id"])
            break
    if type_str != "legal":
        obj["type_str"] = type_str
        # print(obj["id"])
        return obj
    all_equation_layers = convert_equations(equations)
    # print()
    obj["equation_layer"]= all_equation_layers
    process_all_question(all_equation_layers)
    return obj

def process_file(file:str, out_file:str):
    data = read_data(file=file)
    invalid_ans = 0
    total_legal = 0
    for idx, obj in tqdm(enumerate(data), desc=f"Reading {file}", total=len(data)):
        ans = process_options_and_answers(obj["options"].strip(), obj["correct"].strip())
        obj["answer"] = float(ans) if ans is not None else None
        obj = process_obj(obj)
        if ans is None:
            invalid_ans +=1
            obj["type_str"] = "invalid_answer"
        total_legal += 1 if obj["type_str"] == "legal" else 0
    print(f"number of invalid ans: {invalid_ans}, total legal: {total_legal} out of all: {len(data)}")
    # write_data(file=out_file, data=data)
    print(const_list)
    sorted_counter = sorted(const2num.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(sorted_counter)
    print([k[0] for k in sorted_counter])
    const_list.clear()
    const2num.clear()

def get_stat(file:str):
    data = read_data(file=file)
    argnum2num = Counter()
    equation_num = Counter()
    num_valid = 0
    unique_operations = set()
    subject2op = defaultdict(set)
    for obj in tqdm(data, desc=f"Reading {file}", total=len(data)):
        linear_formula = obj["linear_formula"]
        equations = linear_formula.split("|")
        equations = [eq for eq in equations if eq.strip() != '']
        equation_num[len(equations)] += 1
        valid = True
        for equation in equations:
            eles = equation.split(",")
            op_name = eles[0].split("(")[0]
            if len(eles) != 1 and len(eles) !=2:
                valid = False
            if op_name in filtered_ops:
                valid = False
            else:
                if op_name == '':
                    print(obj)
                unique_operations.add(op_name)
                subject2op[obj['category']].add(op_name)
            argnum2num[len(eles)] +=1
        type_str = "legal" if valid else "illegal"
        num_valid += 1 if valid else 0
    print(f"Total numner: {len(data)}")
    print(f"equation num: {equation_num}")
    print(f"arg num num: {argnum2num}")
    print(f" valid  num {num_valid} out of total: {len(data)}")
    print(f"unique operations num: {len(unique_operations)}\n operations: {unique_operations}")
    # for k in subject2op:
    #     print(k, subject2op[k])



if __name__ == '__main__':
    const_list = set()
    const2num = Counter()
    # process_file("data/MathQA/train.json", "data/MathQA/mathqa_train_nodup.json")
    # process_file("data/MathQA/dev.json", "data/MathQA/mathqa_dev_nodup.json")
    # process_file("data/MathQA/test.json", "data/MathQA/mqthqa_test_nodup.json")

    process_file("data/MathQA/train_tan_filtered.json", "data/MathQA/mathqa_train_nodup.json")
    process_file("data/MathQA/dev_tan_filtered.json", "data/MathQA/mathqa_dev_nodup.json")
    process_file("data/MathQA/test_tan_filtered.json", "data/MathQA/mathqa_test_nodup.json")
    # process_file("data/MathQA/debug.json")
    # get_stat("data/MathQA/train.json")
    # get_stat("data/MathQA/dev.json")
    # get_stat("data/MathQA/test.json")

    # x = {'1.6', '26.0', '3.1416', '3.6', '0.3937', '12.0', '0.25', '10.0', '100.0', '4.0', '1000.0', '6.0', '1.0', '5.0', '0.2778', '0.33', '3600.0', '180.0', '0.5', '60.0', '2.0', '3.0', '360.0', '52.0'}
    # y = {'3.1416', '3.6', '12.0', '0.25', '10.0', '100.0', '4.0', '1000.0', '6.0', '1.0', '5.0', '0.2778', '3600.0', '180.0', '0.5', '60.0', '2.0', '3.0', '360.0'}
    # z = {'3.1416', '3.6', '12.0', '0.25', '10.0', '100.0', '4.0', '1000.0', '6.0', '1.0', '5.0', '0.2778', '0.33', '3600.0', '180.0', '0.5', '60.0', '2.0', '3.0', '360.0', '52.0'}
    # x.update(y)
    # x.update(z)
    # print(x)
    # print(len(x))