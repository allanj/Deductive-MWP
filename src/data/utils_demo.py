from typing import List, Union, Dict
import pprint
from src.eval.utils import compute_value_for_parallel_equations

uni_labels = [
    '+','-', '-_rev', '*', '/', '/_rev'
]

def get_label_ids_parallel(constant2id: Dict, parallel_equation_layers: List[List], add_replacement: bool) -> Union[
    List[List[int]], None]:
    # in this data, only have one or zero bracket
    label_ids = []
    num_constant = len(constant2id) if constant2id is not None else 0
    accumulate_eqs = [0]
    for p_idx, equation_layers in enumerate(parallel_equation_layers):
        label_ids.append([])
        for l_idx, layer in enumerate(equation_layers):
            left_var, right_var, op = layer
            if left_var == right_var and (not add_replacement):
                return None
            is_stop = 1 if l_idx == len(equation_layers) - 1 and p_idx == len(parallel_equation_layers) - 1 else 0
            if (not left_var.startswith("m_")):
                if constant2id is not None and left_var in constant2id:
                    left_var_idx = constant2id[left_var] + accumulate_eqs[p_idx]
                else:
                    # try:
                    assert ord(left_var) >= ord('a') and ord(left_var) <= ord('z')
                    # except:
                    #     print("seohting")
                    left_var_idx = (ord(left_var) - ord('a') + num_constant + accumulate_eqs[p_idx])
            else:
                _, m_p_idx, m_v_idx = left_var.split("_")
                m_p_idx, m_v_idx = int(m_p_idx), int(m_v_idx)
                # left_var_idx = -1
                left_var_idx = accumulate_eqs[p_idx] - accumulate_eqs[m_p_idx] - m_v_idx - 1
            if (not right_var.startswith("m_")):
                if constant2id is not None and right_var in constant2id:
                    right_var_idx = constant2id[right_var] + accumulate_eqs[p_idx]
                else:
                    assert ord(right_var) >= ord('a') and ord(right_var) <= ord('z')
                    right_var_idx = (ord(right_var) - ord('a') + num_constant + accumulate_eqs[p_idx])
            else:
                _, m_p_idx, m_v_idx = right_var.split("_")
                m_p_idx, m_v_idx = int(m_p_idx), int(m_v_idx)
                right_var_idx = accumulate_eqs[p_idx] - accumulate_eqs[m_p_idx] - m_v_idx - 1
            # try:
            assert right_var_idx >= 0
            # except:
            #     print("right var index error")
            #     return "right var index error"
            # try:
            assert left_var_idx >= 0
            # except:
            #     return "index error"

            if left_var.startswith("m_") or right_var.startswith("m_"):
                if left_var.startswith("m_") and (not right_var.startswith("m_")):
                    assert left_var_idx < right_var_idx
                    op_idx = uni_labels.index(op)
                    label_ids[p_idx].append([left_var_idx, right_var_idx, op_idx, is_stop])
                elif not left_var.startswith("m_") and right_var.startswith("m_"):
                    assert left_var_idx > right_var_idx
                    op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                    label_ids[p_idx].append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    ## both starts with m
                    if left_var_idx >= right_var_idx:  ##left larger means left m_idx smaller
                        op = op[:-4] if left_var_idx == right_var_idx and op.endswith("_rev") else op
                        op_idx = uni_labels.index(op)
                        if left_var_idx > right_var_idx and (op not in ["+", "*"]):
                            op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(
                                op[:-4])
                        label_ids[p_idx].append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        # left < right
                        if (op in ["+", "*"]):
                            op_idx = uni_labels.index(op)
                            label_ids[p_idx].append([left_var_idx, right_var_idx, op_idx, is_stop])
                        else:
                            # assert not op.endswith("_rev")
                            assert "+" not in op and "*" not in op
                            op_idx = uni_labels.index(op)
                            label_ids[p_idx].append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                if left_var_idx <= right_var_idx:
                    if left_var_idx == right_var_idx and op.endswith("_rev"):
                        op = op[:-4]
                    op_idx = uni_labels.index(op)
                    label_ids[p_idx].append([left_var_idx, right_var_idx, op_idx, is_stop])
                else:
                    # left > right
                    if (op in ["+", "*"]):
                        op_idx = uni_labels.index(op)
                        label_ids[p_idx].append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        # assert not op.endswith("_rev")
                        op_idx = uni_labels.index(op + "_rev") if not op.endswith("_rev") else uni_labels.index(op[:-4])
                        label_ids[p_idx].append([right_var_idx, left_var_idx, op_idx, is_stop])
        accumulate_eqs.append(accumulate_eqs[len(accumulate_eqs) - 1] + len(equation_layers))
    return label_ids


if __name__ == '__main__':
    constant2id = {"1": 0, "PI": 1}
    parallel_equations = [
        [
            ['a', 'b', '*'],
            ['d', 'e', '+']
        ],
        [
            ['m_0_0', 'c', '+'],
            ['m_0_1', 'f', '+']
        ],
        [
            ['m_0_1', 'm_1_1', '+']
        ]

    ]
    res = get_label_ids_parallel(constant2id=constant2id, parallel_equation_layers=parallel_equations, add_replacement=True)
    num_list = [9,8,10,4,3,6]
    value = compute_value_for_parallel_equations(parallel_equations=res,
                                         num_list=num_list,
                                         num_constant=2,
                                         constant_values=[1.0, 3.14])
    print(value)
    pp = pprint.PrettyPrinter(width=20, compact=False)
    pp.pprint(res)