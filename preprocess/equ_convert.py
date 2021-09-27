from copy import deepcopy
from tqdm import tqdm
from src.utils import read_data, write_data

class EquationCoverter:
    # this is a converter for mwp equation calculation
    def __init__(self, mid_equ=None, post_equ=None, var_num_map=None, equation_layer=None, value=None, constant=None):
        self.mid_equation = mid_equ #  ['(','temp_a','+','temp_b',')','/','temp_c']
        self.post_equation = post_equ  # ['temp_a', 'temp_b', +, 'temp_c', '/']
        self.var_num_map = var_num_map # {temp_a: '10', temp_b:'7', temp_c:'5'}
        self.equation_layer = equation_layer
        self.value = value
        self.op = ['^','+','-','*','/']
        self.constant_dict = constant
        if constant is not None and self.var_num_map is not None:
            self.var_num_map.update(constant)

    def init_from_mid(self, mid_equ):
        self.mid_equation = mid_equ
        self.post_equation = self.mid2post()
        _ = self.obtain_m1()

    def init_from_post(self, post_equ):
        self.post_equation = post_equ
        _ = self.obtain_m1()

    def mid2post(self, equ_list=None):
        if equ_list is None: equ_list = self.mid_equation
        stack = []
        post_equ = []
        op_list = self.op
        priori = {'^': 3, '*': 2, '/': 2, '+': 1, '-': 1}
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
                # if elem == 'PI':
                #    post_equ.append('3.14')
                # else:
                #    post_equ.append(elem)
                post_equ.append(elem)
        while stack != []:
            post_equ.append(stack.pop())
        self.post_equation = post_equ
        return post_equ

    def count_op(self, one_side):
        count = 0
        for ele in one_side:
            if ele in self.op: count += 1
        return count

    def find_pos_all_layers_before(self, one_side, layers):
        for i in range(len(layers)):
            layer = layers[i]
            for j in range(len(layer)):
                if one_side == layer[j]: return i,j
        else: return 'unfound'

    def obtain_m1(self, equation=None):
        if equation is None: equation = self.post_equation
        # a*b*c*d
        cal_stack = []
        m_list = []
        layers = [[] for i in range(20)]
        this_layer_ref = [[] for i in range(20)]

        #indexes = dict(zip(list1, list(range(len(list1)))))
        #this_layer = 0
        for ele in equation:
            if ele not in self.op:
                cal_stack.append([ele, -1])
            else:
                right = cal_stack.pop()
                left = cal_stack.pop()
                #print(cal_stack)
                right, rl = right[0], right[1]
                left, ll = left[0], left[1]
                # layer depends on the largest length of left and right
                this_layer = max(rl, ll) + 1 # if no op, then layer 0
                if this_layer > 0: # need to use ref
                    if self.count_op(left) > 0:
                        left_layer, left_pos = self.find_pos_all_layers_before(left, layers[:this_layer])
                        left_string = str(left_layer)+'_'+str(left_pos)
                    else: left_string = left
                    if self.count_op(right) > 0:
                        right_layer, right_pos = self.find_pos_all_layers_before(right, layers[:this_layer])
                        right_string = str(right_layer)+'_'+str(right_pos)
                    else: right_string = right
                    if left_string+' '+ele+' '+right_string not in this_layer_ref[this_layer]:
                        this_layer_ref[this_layer].append(left_string+' '+ele+' '+right_string)
                if left+' '+ele+' '+right not in layers[this_layer]:
                    layers[this_layer].append(left+' '+ele+' '+right)

                # find in the previous layer
                m_list.append(left+' '+ele+' '+right)
                cal_stack.append([left+' '+ele+' '+right, this_layer])
        #print(m_list, layers)
        this_layer_ref[0] = layers[0]
        for i in range(len(this_layer_ref)):
            if len(this_layer_ref[i]) == 0:
                this_layer_ref = this_layer_ref[:i]
                break
        self.equation_layer =  this_layer_ref
        return deepcopy(this_layer_ref)

    def eqLayer2value(self, eqlayers=None, value_map=None):
        if eqlayers is None:
            eqlayers = deepcopy(self.equation_layer)
        if value_map is None:
            value_map = self.var_num_map
        # # [['a/b', 'a/c'], ['00*d'], ['a-10'], ['20/01']]
        debug_eqlayers = deepcopy(eqlayers)
        def numeric(one_side):
            if str(one_side) in value_map: return value_map[one_side]
            else:
                one_side = one_side.split('_')
                return eqlayers[int(one_side[0])][int(one_side[1])]

        for i in range(len(eqlayers)):
            for j in range(len(eqlayers[i])):
                eqlayers[i][j] = eqlayers[i][j].split()
                left, op, right = eqlayers[i][j][0], eqlayers[i][j][1], eqlayers[i][j][2]
                left = numeric(left)
                right = numeric(right)
                if op == '^': op = '**'
                eqlayers[i][j] = str(eval(left+op+right))
        return float(eqlayers[-1][0])

    def eqLayer2equation(self, eqlayer):
        pass

    def post2value(self, post_equ=None, val_num_map=None):
        if post_equ is None: post_equ = self.post_equation
        cal_stack = []
        if val_num_map is None:
            val_num_map = self.var_num_map
        num_equ = []
        for ele in post_equ:
            if ele not in self.op:
                num_equ.append(val_num_map[ele])
            else:num_equ.append(ele)
        for ele in num_equ:
            if ele not in self.op:
                cal_stack.append(ele)
            else:
                right = cal_stack.pop()
                left = cal_stack.pop()
                if ele == '^': ele = '**'
                cal_stack.append(str(eval(left+ele+right)))
        return float(cal_stack[0])



#print(EquationCoverter.postfix_equation(['(','temp_a','+','temp_b',')','/','temp_c']))
def an_example():
    #mid_equ = "( ( temp_c + temp_a + temp_b ) * temp_b - temp_a ) * temp_b"
    #mid_equ = "( a - a / b * d ) / ( a / c )"
    mid_equ = "temp_d / ( temp_c - temp_b ) * ( temp_a - temp_c )"
    converter = EquationCoverter(mid_equ=mid_equ.split(),
                                 var_num_map={'temp_a': '10', 'temp_b':'7', 'temp_c':'5', 'temp_d':'2'})
    post_equ = converter.mid2post()
    value_1 = converter.post2value()
    equ_layer = converter.obtain_m1()
    value_2 = converter.eqLayer2value()
    print(post_equ, value_1, equ_layer, value_2, sep='\n')

#an_example()

def load23k(file, output_file= None):
    data = read_data(file)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    equ_layer = []
    invalid_number = 0
    for point in tqdm(data, desc="processing file", total=len(data)):
        equ = point["target_template"][2:]
        for i in range(len(equ)):
            if equ[i][0] == 't': equ[i] = equ[i][-1]
        #for i in range(len(equ)): eq
        num_map = dict(zip(alphabet[:len(point["num_list"])], [str(ele) for ele in point["num_list"]]))
        converter = EquationCoverter(var_num_map=num_map, constant={'1':'1','PI':'3.14'})
        try:
            converter.init_from_mid(equ)
            if len(equ) != 1:
                ans = converter.eqLayer2value()
                # print('cal_ans:',ans, '  ||real:',point['ans'])
                point['ans'] = point['ans'].replace('%', '/100')
                # try:
                #     if abs(ans - eval(point['ans'])) > 1e-6:
                #         print(point['id'], ':', converter.equation_layer, point['equation'], ans,
                #                                                    point['ans'], point['num_list'])
                #         print(point['text'],'\n')
                # except:
                #     pass
        except:
            print(point["target_template"])
            print(point)
            print('wrong equation', equ)
            print(equ)
            invalid_number += 1
        point["parallel_equation_layer"] = converter.equation_layer
        equ_layer.append(converter.equation_layer)
    write_data(file=output_file, data= data)
    print(f"invalid number: {invalid_number}")

load23k("../data/math23k/valid23k_processed.json", "../data/math23k/valid23k_processed_parallel.json")
load23k("../data/math23k/train23k_processed.json", "../data/math23k/train23k_processed_parallel.json")
load23k("../data/math23k/test23k_processed.json", "../data/math23k/test23k_processed_parallel.json")




