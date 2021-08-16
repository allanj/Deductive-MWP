

fv_labels = [
    '0 1 2 +', # v0 + v1 = v2 (without 3)
    '0 2 1 +',
    '1 2 0 +',
    '0 1 2 *',
    '0 2 1 *',
    '1 2 0 *',
    '1 2 3 +', # v0 + v1 = v2 (without 0)
    '1 3 2 +',
    '2 3 1 +',
    '1 2 3 *',
    '1 3 2 *',
    '2 3 1 *',
    '0 2 3 +', # v0 + v1 = v2 (without 1)
    '0 3 2 +',
    '2 3 0 +',
    '0 2 3 *',
    '0 3 2 *',
    '2 3 0 *',
    '0 1 3 +', # v0 + v1 = v2 (without 2)
    '0 3 1 +',
    '1 3 0 +',
    '0 1 3 *',
    '0 3 1 *',
    '1 3 0 *',
]


pretty_labels = [
    'V0 + V1 = V2', # v0 + v1 = v2
    'V0 + V2 = V1',
    'V1 + V2 = V0',
    'V0 * V1 = V2',
    'V0 * V2 = V1',
    'V1 * V2 = V0',
    'V1 + V2 = V3', # v0 + v1 = v2
    'V1 + V3 = V2',
    'V2 + V3 = V1',
    'V1 * V2 = V3',
    'V1 * V3 = V2',
    'V2 * V3 = V1',
    'V0 + V2 = V3', # v0 + v1 = v2
    'V0 + V3 = V2',
    'V2 + V3 = V0',
    'V0 * V2 = V3',
    'V0 * V3 = V2',
    'V2 * V3 = V0',
    'V0 + V1 = V3', # v0 + v1 = v2
    'V0 + V3 = V1',
    'V1 + V3 = V0',
    'V0 * V1 = V3',
    'V0 * V3 = V1',
    'V1 * V3 = V0'
]

from typing import List

def check_valid_labels(mapped_equation, uni_labels: List[str]):
    if len(mapped_equation) - len(mapped_equation.replace("(", "")) > 1:
        return False
    assert mapped_equation.startswith("x=")
    mapped_equation = ' '.join(mapped_equation[2:].split(' '))
    labels = []
    if len(mapped_equation) - len(mapped_equation.replace("(", "")) == 1:
        pass
    else:
        ## find * /, first operation
        var_and_ops = mapped_equation.split(" ")
        first_prioritize_operation_index = None
        for i in range(1, len(var_and_ops), 2):
            assert var_and_ops[i] in ["+", "-", "*", "/"]
            if var_and_ops[i] in ["*", "/"]:
                first_prioritize_operation_index = i
                break
        if first_prioritize_operation_index is None:
            for i in range(1, len(var_and_ops), 2):
                assert var_and_ops[i] in ["+", "-"]
                op = var_and_ops[i]
                left_var_index = ord(var_and_ops[i - 1][-1]) - ord('a') if i == 1 else -1
                right_var_index = ord(var_and_ops[i + 1][-1]) - ord('a')
                if left_var_index == right_var_index:
                    print("found left right same id")
                    return False
                actual_op_id = uni_labels.index(op) if left_var_index < right_var_index else uni_labels.index(op + "_rev")
                labels.append((left_var_index, right_var_index, uni_labels[actual_op_id]))
        else:
            op = var_and_ops[first_prioritize_operation_index]
            left_var_index = ord(var_and_ops[first_prioritize_operation_index - 1][-1]) - ord('a')
            right_var_index = ord(var_and_ops[first_prioritize_operation_index + 1][-1]) - ord('a')
            actual_op_id = uni_labels.index(op) if left_var_index < right_var_index else uni_labels.index(op + "_rev")
            labels.append((left_var_index, right_var_index, uni_labels[actual_op_id]))
            mapped_equation = mapped_equation.replace(
                f"{var_and_ops[first_prioritize_operation_index - 1]} {op} {var_and_ops[first_prioritize_operation_index+1]}", "m0"
            )
            var_and_ops = mapped_equation.split(" ")
            ## starting from the right hand side.
            m0_idx = -1
            start_idx = -1
            for i, ele in enumerate(var_and_ops):
                if ele == "m0":
                    m0_idx = i
                    continue
                if m0_idx != -1 and ele in ["*", "/"]:
                    if i != m0_idx + 1:
                        print("still have * / operation after the first m0")
                        return False
                    else:
                        start_idx = m0_idx + 1
            assert m0_idx != -1
            if start_idx != -1:
                for i in range(start_idx, len(var_and_ops), 2):
                    op = var_and_ops[i]
                    right_var_index = ord(var_and_ops[i + 1][-1]) - ord('a')
                    actual_op_id = uni_labels.index(op)
                    labels.append((-1, right_var_index, uni_labels[actual_op_id]))
            for i in range(m0_idx - 1, 0, -2):
                op = var_and_ops[i]
                right_var_index = ord(var_and_ops[i - 1][-1]) - ord('a')
                actual_op_id = uni_labels.index(op + "_rev")##beacuase m0 is actually on the right
                labels.append((-1, right_var_index, uni_labels[actual_op_id]))
        return labels


if __name__ == '__main__':
    check_valid_labels("", [
        '+','-', '-_rev', '*', '/', '/_rev'
    ])