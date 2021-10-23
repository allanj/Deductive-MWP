import json
with open("../data/mwp_raw.json", 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
# special num: {pi - 圆; 1 - ratio; 24 - time}
exceeding_ele_recorder = {}

def num_type():
    cate = {'.': '小数', '/': '分数', ':': '比例/时间'}

def num_step(equation):
    calc = ['^','/','*','+','-']
    equation = equation.replace('**','^')
    count = 0
    for ele in equation:
        if ele in calc: count += 1
    return count

def appear_times(equation, variables, num_list):
    for i in range(len(variables)):
        variables[i]['used_times'] = equation.count(variables[i]['name'])
        variables[i]['num_value'] = num_list[i]
    return variables

def text_spliter(text):
    # cut the full text into spans, while locate which is which
    # if a span is too short, then connect the short spans along with their head and tail
    split_tokens = [",", "?", "。", "，", "？", "；", ";"]
    spans = []
    last = 0
    for i in range(len(text)):
        if text[i] in split_tokens:
            spans.append(text[last:i])
            last = i + 1
        elif text[i] == '.':
            if 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                continue
            else:
                spans.append(text[last:i])
                last = i + 1
    # TODO: to join short spans together

def ob_variable(processed_data):
    pass
    # variable = {
    #     'name':,
    #     'span_start_pos':,
    #     'span_end_pos':,
    #     'pos':,
    #     'value':,
    #     'type':,
    #     'is_used':,
    #     'span_text':,
    # }

def check_length(raw_data): return (len(raw_data))

def postfix_equation(equ_list, id):
    #ipdb.set_trace()
    stack = []
    post_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priori = {'^':3, '*':2, '/':2, '+':1, '-':1}
    for elem in equ_list:
        elem = str(elem)
        if elem == '(':
            stack.append('(')
        elif elem == ')':
            while 1:
                try:
                    op = stack.pop()
                except:
                    print(equ_list)
                if op == '(':
                    break
                else:
                    post_equ.append(op)
        elif elem in op_list:
            while 1:
                if stack == []:
                    break
                elif stack[-1] == '(':
                    break
                elif priori[elem] > priori[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    post_equ.append(op)
            stack.append(elem)
        else:
            #if elem == 'PI':
            #    post_equ.append('3.14')
            #else:
            #    post_equ.append(elem)
            post_equ.append(elem)
    while stack != []:
        post_equ.append(stack.pop())
    return post_equ

def avg_seq_length(raw_data):
    len_tot, char_len = 0, 0
    for mwp in raw_data:
        char_len +=  len(mwp["original_text"])
        seq = mwp["segmented_text"].split()
        len_tot += len(seq)
    return len_tot/len(raw_data), char_len/len(raw_data)

def fraction_replace(segmented_text):
    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        segmented_text = segmented_text.replace(' / '+digit, '/'+digit).replace(' : '+digit, ':'+digit)
    return segmented_text

def text_process(raw_data):
    count = 0
    processed_data = []
    max_raw, max_process, avg_raw, avg_process, count_repeat, count_process_repeat, count_equ_repeat = 0, 0, 0, 0, 0, 0, 0
    for i in range(len(raw_data)):
        elem = raw_data[i]
        #if elem['id'] == '158277508' or elem['id'] == '155739610': continue
        text = fraction_replace(elem["segmented_text"]).split()
        mapped_text, quantity_map, variables = text_map(text)
        max_raw = max(max_raw, len(quantity_map))
        avg_raw += len(quantity_map)
        norm_equ = equ_map(elem["equation"], quantity_map, elem["id"])
        if len(set(quantity_map)) != len(quantity_map): count_repeat += 1
        if norm_equ is None:
            count += 1
            continue
        else:
            t_set = []
            for ele in norm_equ:
                if ele[0] == 't': t_set.append(ele)
            if len(set(t_set)) != len(t_set):
                repeated = True
                count_equ_repeat += 1
            else: repeated = False
            max_process = max(max_process, len(quantity_map))
            avg_process += len(quantity_map)
            processed_data.append(raw_data[i])
            if len(set(quantity_map)) != len(quantity_map):
                count_process_repeat += 1
                #print(elem['id'])
            processed_data[-1]['sentence'] = ' '.join(text)
            #processed_data[-1]['mapped_text'] = ' '.join(mapped_text).replace("temp_","@")
            mapped_text = ' '.join(mapped_text).replace("temp_","@ ")
            processed_data[-1]['mapped_text'] = mapped_text
            mapped_text = mapped_text.split()
            var_pos = []
            for index in range(len(mapped_text)):
                if mapped_text[index] == '@': var_pos.append(index)
            processed_data[-1]['position_list'] = var_pos
            processed_data[-1]['posted_equation'] = ' '.join(postfix_equation(norm_equ, processed_data[-1]['id']))
            norm_equ = ' '.join(norm_equ)
            if norm_equ.find('PI')==-1 and norm_equ.find('1') == -1: processed_data[-1]['legal'] = True
            else: processed_data[-1]['legal'] = False
            processed_data[-1]['mapped_equation'] = "x= " + norm_equ
            processed_data[-1]['num_list'] = num_convert(quantity_map, elem['id'])
            processed_data[-1]['num_steps'] = num_step(norm_equ)
            processed_data[-1]['variables'] = appear_times(norm_equ, variables, processed_data[-1]['num_list'])
            processed_data[-1]['repeated'] = repeated


    with open("../data/large_math/mwp_processed.json", "w", encoding='utf-8') as f1:
        json.dump(processed_data, f1,ensure_ascii=False, indent=4)
    print("count", count)
    print("count_repeat", count_equ_repeat)
    print("how many datapoints to use", len(processed_data))
    print("word per seq, char per seq:",avg_seq_length(processed_data))
    print("max_var_num", max_raw, max_process, "repeated var",count_repeat, count_process_repeat)

def is_digit(chara):return chara in ['0','1','2','3','4','5','6','7','8','9']

def num_convert(quantity_map, id):
    num_list = []
    for token in quantity_map:
        if '/' in token:
            token = token.split('/')
            num_list.append(float(token[0])/float(token[1]))
        elif ':' in token:
            token = token.split(':')
            if float(token[1])!= 0: num_list.append(float(token[0])/float(token[1]))
            else: num_list.append(-9999)
        elif token[-1] == '%':
            num_list.append(float(token[:-1])/100)
        else:
            num_list.append(float(token))
    return num_list

def text_map(text):
    #TODO: get the rate/fraction/time right
    quantities, type = [], []
    quant_dict = ["temp_a", "temp_b", "temp_c", "temp_d", "temp_e", "temp_f", "temp_g", "temp_h",
    "temp_i", "temp_j", "temp_k", "temp_l", "temp_m", "temp_n", "temp_o", "temp_q", "temp_r","temp_s", "temp_t" ]
    mapped_text, variables = [], []
    #text = text.replace(' / ','/').replace(' : ',':')
    split_tokens = [",", "?", "。", "，", "？", "；", ";",'.']
    spans = {}
    last = 0

    for i in range(len(text)):
        token = text[i]
        if text[i] in split_tokens:
            spans[last] = [text[last:i],i]
            last = i + 1
        elif i == len(text) -1 and last not in spans.keys():
            spans[last] = [text[last:], i]

        if is_digit(token[0]):
            name = quant_dict[len(quantities)]
            mapped_text.append(name)
            if token[-1] == '%':
                quantities.append(str(float(token[:-1])/100))
            else:
                quantities.append(token)
        else:
            for chars in token:
                mapped_text.append(chars)
            continue

        if token[-1] == "%":
            type.append('百分数')  # percentage
        elif len(token) > 1 and token.find("/") != -1:
            type.append('分数')  # fraction
        elif len(token) > 1 and token.find(':') != -1:
            type.append('时间/比例')
        elif len(token) > 1 and token.find('.') != -1:
            type.append('小数')
        else:
            try:
                float(token)
            except: print(token)
            type.append('基数')  # just int number

        variable = {
            'name':name,
            'span_start_pos':last,
            'span_end_pos':None,
            'pos':len(mapped_text)-1,
            'token':token,
            'type':type[-1],
            'used_times':None,
            'span_text':None,
        }
        variables.append(variable)

    for i in range(len(variables)):
        variable = variables[i]
        belong_span = ' '.join(spans[variable['span_start_pos']][0])
        end_pos = spans[variable['span_start_pos']][1]
        span_text = belong_span.replace(variable['token'], '<quant>')
        variables[i]['span_end_pos'] = end_pos
        variables[i]['span_text'] = span_text

    return mapped_text, quantities, variables

def mapping_template(quantity_map, quantity):
    quant_dict = ["temp_a", "temp_b", "temp_c", "temp_d", "temp_e", "temp_f", "temp_g", "temp_h",
    "temp_i", "temp_j", "temp_k", "temp_l", "temp_m", "temp_n", "temp_o", "temp_q","temp_r","temp_s", "temp_t"]
    if quantity in quantity_map: return quant_dict[quantity_map.index(quantity)]
    elif quantity == '3.14' or quantity == '3.1416': return 'PI'
    elif quantity =='100%' or '1' or quantity == '1.0': return '1'
    elif quantity in ['2', '3', '4', '5', '6', '7', '12','30','31','8','10']: return quantity # AddConstant
    else: return None

def equ_map(equation, quantity_map, id):

    quant_dict = ["temp_a", "temp_b", "temp_c", "temp_d", "temp_e", "temp_f", "temp_g", "temp_h",
    "temp_i", "temp_j", "temp_k", "temp_l", "temp_m", "temp_n", "temp_o", "temp_q", "temp_r","temp_s", "temp_t"]
    operator_list = ['+', '-', '*', '(', ')','^']
    equation = equation.replace(' ','')
    equation = equation.replace('**', '^')
    fraction = 0
    for ele in quantity_map:
        if ele.find('/') != -1: fraction = 1
    if fraction == 0: operator_list.append('/') #无分数
    for ele in operator_list:
        equation = equation.replace(ele, ' ' + ele + ' ')
    equation1 = equation.split()
    norm_equ = []
    # 360/12/ (144/8/3)
    for ele in equation1:
        if ele[-1] == '%': ele = str(float(ele[:-1])/100)
        if ele in quantity_map: norm_equ.append(quant_dict[quantity_map.index(ele)])# this is fraction.
        elif ele in operator_list: norm_equ.append(ele)
        elif '/' in ele:
            ele = ele.split('/') #this is an operator
            if ele[0] != '':
                former = mapping_template(quantity_map, ele[0])
                if former is not None: norm_equ.append(former)
                else:
                    if ele[0] in exceeding_ele_recorder.keys():
                        exceeding_ele_recorder[ele[0]] += 1
                    else:
                        exceeding_ele_recorder[ele[0]] = 0
            norm_equ.append('/')
            if ele[1] != '':
                later = mapping_template(quantity_map, ele[1])
                if later is not None: norm_equ.append(later)
                else:
                    if ele[1] in exceeding_ele_recorder.keys():
                        exceeding_ele_recorder[ele[1]] += 1
                    else:
                        exceeding_ele_recorder[ele[1]] = 0
        elif ele == '3.14' or ele == '3.1416': norm_equ.append('PI')
        elif ele == '100%' or ele == '1' or ele == '1.0': norm_equ.append('1')
        #elif ele in ['1', '2']:
        #    norm_equ.append(ele)
        elif ele in ['2', '3', '4', '5', '6', '7', '12','30','31','8','10']: norm_equ.append(ele) # AddConstant
        else:
            if ele in exceeding_ele_recorder.keys(): exceeding_ele_recorder[ele] += 1
            else: exceeding_ele_recorder[ele] = 0
            return None
    return norm_equ

    # /有两种可能，分数或者运算
    # 如果前后整数且能找得到对应，那就是分数


print("total data mun: ", check_length(raw_data))
print("word_per_seq, char_per_seq: ", avg_seq_length(raw_data))
'''

quantity type: time, float, int, ratio, percentage, fraction
'''

text_process(raw_data)

exceeding_ele_recorder_sorted = sorted(exceeding_ele_recorder.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
print("constant dict",exceeding_ele_recorder_sorted)

# def num_mapping(text):
#     rpt_count = 0
#     repeated_ids = []
#     for mwp in raw_data:
#         seq = mwp["segmented_text"].split()
#         num_stack, pos_stack = num_locater(seq)
#         if len(set(num_stack)) != len(num_stack):
#             rpt_count += 1
#             repeated_ids.append(mwp["id"])
#     print("repeated num cases", rpt_count)
#     return repeated_ids
#
# def num_locater(seq):
#     num_stack = []
#     pos_stack = []
#     for i in range(len(seq)):
#         word = seq[i]
#         if word[0].isdigit():
#             num_stack.append(word)
#             pos_stack.append(i)
#     return num_stack, pos_stack

def bad_cases(raw_data):
    with open("wrong_ids.json") as wo:
        wrong_cases = json.load(wo)
    count = 0
    processed_data = []

    count_repeat, count_process_repeat, count_wrong_repeat, avg_quanti_num = 0, 0, 0, 0
    for i in range(len(raw_data)):
        elem = raw_data[i]
        text = fraction_replace(elem["segmented_text"]).split()
        mapped_text, quantity_map = text_map(text)
        norm_equ = equ_map(elem["equation"], quantity_map, elem["id"])
        if elem['id'] in wrong_cases: avg_quanti_num += len(quantity_map)
        if len(set(quantity_map)) != len(quantity_map):
            count_repeat += 1
            if elem['id'] in wrong_cases: count_wrong_repeat += 1
        if norm_equ is None: count += 1
        else:
            processed_data.append(raw_data[i])
            if len(set(quantity_map)) != len(quantity_map):
                count_process_repeat += 1
                #print(elem['id'])
            processed_data[-1]['mapped_text'] = mapped_text
            processed_data[-1]['mapped_equation'] = ["x", "="]  + norm_equ
            processed_data[-1]['posted_equation'] = postfix_equation(processed_data[-1]['mapped_equation'], processed_data[-1]['id'])
            processed_data[-1]['num_list'] = num_convert(quantity_map, elem['id'] )

    print(count)
    print(len(processed_data))
    print(avg_seq_length(processed_data))
    print(count_repeat, count_process_repeat, count_wrong_repeat, avg_quanti_num/len(wrong_cases))
