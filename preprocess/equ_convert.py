from copy import deepcopy
class EquationCoverter:
    # this is a converter for mwp equation calculation
    def __init__(self, mid_equ=None, post_equ=None, var_num_map=None, equation_layer=None, value=None):
        self.mid_equation = mid_equ #  ['(','temp_a','+','temp_b',')','/','temp_c']
        self.post_equation = post_equ  # ['temp_a', 'temp_b', +, 'temp_c', '/']
        self.var_num_map = var_num_map # {temp_a: '10', temp_b:'7', temp_c:'5'}
        self.equation_layer = equation_layer
        self.value = value
        self.op = ['^','+','-','*','/']

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
            if ele[0] == 't':
                cal_stack.append([ele[-1], -1])
            else:
                right = cal_stack.pop()
                left = cal_stack.pop()
                right, rl = right[0], right[1]
                left, ll = left[0], left[1]
                # layer depends on the largest length of left and right
                this_layer = max(rl, ll) + 1 # if no op, then layer 0
                if this_layer > 0: # need to use ref
                    if self.count_op(left) > 0:
                        left_layer, left_pos = self.find_pos_all_layers_before(left, layers[:this_layer])
                        left_string = str(left_layer)+str(left_pos)
                    else: left_string = left
                    if self.count_op(right) > 0:
                        right_layer, right_pos = self.find_pos_all_layers_before(right, layers[:this_layer])
                        right_string = str(right_layer)+str(right_pos)
                    else: right_string = right
                    this_layer_ref[this_layer].append(left_string+ele+right_string)
                layers[this_layer].append(left+ele+right)

                # find in the previous layer
                m_list.append(left+ele+right)
                cal_stack.append([left+ele+right, this_layer])
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
            eqlayers = self.equation_layer
        if value_map is None:
            value_map = self.var_num_map
        # # [['a/b', 'a/c'], ['00*d'], ['a-10'], ['20/01']]

        def replace(var):
            for c in range(len(var)):
                if str(var[c]) in self.op:
                    return var[:c], var[c], var[c+1:]

        def numeric(one_side):
            if str(one_side).islower(): return value_map['temp_'+one_side]
            else:
                return eqlayers[int(one_side[0])][int(one_side[1])]

        for i in range(len(eqlayers)):
            for j in range(len(eqlayers[i])):
                left, op, right = replace(eqlayers[i][j])
                left = numeric(left)
                right = numeric(right)
                if op == '^': op = '**'
                eqlayers[i][j] = str(eval(left+op+right))
        return eqlayers[-1][0]

    def eqLayer2equation(self, eqlayer):
        pass

    def post2value(self, post_equ=None):
        if post_equ is None: post_equ = self.post_equation
        cal_stack = []
        assert self.var_num_map is not None
        num_equ = []
        op_list = ['+', '-', '*', '/', '^']
        for ele in post_equ:
            if ele[0] == 't':
                num_equ.append(self.var_num_map[ele])
            else:num_equ.append(ele)
        for ele in num_equ:
            if ele not in op_list:
                cal_stack.append(ele)
            else:
                right = cal_stack.pop()
                left = cal_stack.pop()
                cal_stack.append(str(eval(left+ele+right)))
        return float(cal_stack[0])



#print(EquationCoverter.postfix_equation(['(','temp_a','+','temp_b',')','/','temp_c']))
def an_example():
    #mid_equ = "( ( temp_c + temp_a + temp_b ) * temp_b - temp_a ) * temp_b"
    #mid_equ = "( temp_a - temp_a / temp_b * temp_d ) / ( temp_a / temp_c )"
    mid_equ = "temp_d / ( temp_c - temp_b ) * ( temp_a - temp_c )"
    converter = EquationCoverter(mid_equ=mid_equ.split(),
                                 var_num_map={'temp_a': '10', 'temp_b':'7', 'temp_c':'5', 'temp_d':'2'})
    post_equ = converter.mid2post()
    value_1 = converter.post2value()
    equ_layer = converter.obtain_m1()
    value_2 = converter.eqLayer2value()

    print(post_equ, value_1, equ_layer, value_2, sep='\n')

an_example()