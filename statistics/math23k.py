
from src.utils import read_data, write_data
from  collections import Counter




def get_variable_to_num(file):
    variable_num2inst_num = Counter()
    data = read_data(file=file)
    operation_num2inst_num = Counter()
    for obj in data:
        target_norm_post_template = obj["target_norm_post_template"]
        var_set = set()
        var_list = []
        for word in target_norm_post_template:
            if word.startswith("temp"):
                var_set.add(word)
                var_list.append(word)
        if len(var_list) == 21:
            # print(target_norm_post_template)
            print(obj["original_text"])
        variable_num2inst_num[len(var_set)] += 1
        operation_num2inst_num[len(var_list)] +=1
    print(f"Total number of data: {len(data)}")
    print(f" unique variable num2inst num: {variable_num2inst_num}")
    print(f" operation num2inst num: {operation_num2inst_num}")


def get_opreation2_num(file):
    opnum = Counter()
    data = read_data(file=file)
    res = []
    for obj in data:
        target_norm_post_template = obj["target_norm_post_template"]
        cur_op_num = 0
        for word in target_norm_post_template:
            if word in ["+", "-", "*", "/", "^"]:
                cur_op_num += 1
        opnum[cur_op_num] += 1
        res.append({
            "id": obj["id"],
            "operation_num": cur_op_num
        })
    print(f"Total number of data: opnum; {opnum}")
    write_data("../data/math23k/operation_num.json", res)


if __name__ == '__main__':
    # get_variable_to_num("../data/math23k/whole_processed.json")
    get_opreation2_num("../data/math23k/test23k_processed.json")
    # write_data(file="../data/math23k/whole_processed_convert.json",data=read_data("../data/math23k/whole_processed.json"))