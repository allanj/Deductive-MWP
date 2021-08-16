
from src.utils import read_data, write_data
from  collections import Counter




def get_variable_to_num(file):
    variable_num2inst_num = Counter()
    data = read_data(file=file)
    operation_num2inst_num = Counter()
    total_after_filterd = 0
    for obj in data:
        target_norm_post_template = obj["mapped_equation"].split()
        var_set = set()
        var_list = []
        if not (obj['legal'] and obj['num_steps'] <= 2):
            continue
        for word in target_norm_post_template:
            if word.startswith("temp"):
                var_set.add(word)
                var_list.append(word)
        variable_num2inst_num[len(var_set)] += 1
        operation_num2inst_num[len(var_list)] +=1
        total_after_filterd+=1
        mapped_equation = obj["mapped_equation"]
        for k in range(ord('a'), ord('a')+26):
            mapped_equation = mapped_equation.replace(f"( temp_{chr(k)} )", f"temp_{chr(k)}")
        if len(mapped_equation) - len(mapped_equation.replace("(", "")) > 1:
            print(mapped_equation)
    print(f"Total number of data: {len(data)}, after filtered: {total_after_filterd}")
    print(f" unique variable num2inst num: {variable_num2inst_num}")
    print(f" operation num2inst num: {operation_num2inst_num}")


if __name__ == '__main__':
    get_variable_to_num("../data/complex/mwp_processed.json")
