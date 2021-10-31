
label2op = {
    'Sub': '-',
    'Div': '/',
    'Pow': '**',
    'Mult': '*',
    'Add': '+',
    # 'USub': '-',
}

func2op = {
    'add': 'Add',
    'subtract': 'Sub',
    'multiply': 'Mult',
    'rectangle_area': 'Mult',
    'divide': 'Div',
    'speed': 'Div',
    'power': 'Pow',
    # 'negate': 'USub',
    'inverse': ('Div', 1, 'temp_a'),
    'square_area': ('Pow', 'temp_a', 2),
    'sqrt': ('Pow', 'temp_a', 1 / 2),
    'square_edge_by_area': ('Pow', 'temp_a', 1 / 2),
    'cube_edge_by_volume': ('Pow', 'temp_a', 1 / 2),
    'volume_cube': ('Pow', 'temp_a', 3),
    'surface_cube': [('Pow', 'temp_a', 2), ('Mult', 'm_0', 6)],
    'square_perimeter': ('Mult', 'temp_a', 4),
    'rectangle_perimeter': [('Add', 'temp_a', 'temp_b'), ('Mult', 'm_0', 2)],
    'stream_speed': [('Add', 'temp_a', 'temp_b'), ('Div', 'm_0', 2)],
    'triangle_area': [('Mult', 'temp_a', 'temp_b'), ('Div', 'm_0', 2)],
    'triangle_perimeter': [('Add', 'temp_a', 'temp_b'), ('Add', 'm_0', 'temp_c')],
    'surface_sphere': [('Pow', 'temp_a', 2), ('Mult', 4, 'const_pi'), ('Mult', 'm_0', 'm_1')],
    'volume_sphere': [('Pow', 'temp_a', 3),('Div', 4, 3),  ('Mult', 'm_1', "const_pi"), ('Mult', 'm_0', 'm_2')],
    'rhombus_area': [('Mult', 'temp_a', 'temp_b'), ('Div', 'm_0', 2)],
    'quadrilateral_area': [('Add', 'temp_b', 'temp_c'), ('Mult', 'm_0', 'temp_a'), ('Div', 'm_1', 2)],
    'volume_cylinder': [('Pow', 'temp_a', 2), ('Mult', 'm_0', 'const_pi'),('Mult', 'm_1', 'temp_b')],
    'circle_area': [('Pow', 'temp_a', 2),('Mult', 'm_0', "const_pi") ], # ('Mult', ('Pow', None, 2), "const_pi"),
    'volume_cone': [('Mult', 'temp_a', 'const_pi'), ('Mult', 'm_0', 'temp_b'),('Div', 'm_1',3) ],#('Div', ('Mult', ('Mult', None, 'const_pi'), None), 3),
    'circumface': [('Mult', 'temp_a', 2), ('Mult', 'm_0', "const_pi")], #('Mult', ('Mult', None, 2), "const_pi"),
    'diagonal': [("Pow", 'temp_a', 2),("Pow", 'temp_b', 2), ("Add", 'm_1', 'm_0'), ("Pow", 'm_2', 1/2)],#("Pow", ("Add", ("Pow", None, 2), ("Pow", None, 2)), 1 / 2),
    'volume_rectangular_prism': [('Mult', 'temp_a', 'temp_b'), ('Mult', 'm_0', 'temp_c')],#('Mult', ('Mult', None, None), None),
    'original_price_before_loss': [('Div', 'temp_a', 100), ('Sub', 1, 'm_0'), ('Div', 'temp_b', 'm_1')],#('Div', None, ('Sub', 1, ('Div', None, 100))),
    'original_price_before_gain': [('Div', 'temp_a', 100), ('Add', 'm_0', 1), ('Div', 'temp_b', 'm_1')], #('Div', None, ('Add', 1, ('Div', None, 100))),
    'p_after_gain': [('Div', 'temp_a', 100), ('Add', 'm_0', 1), ('Mult', 'm_1', 'temp_b')], #('Mult', ('Add', 1, ('Div', None, 100)), None),
    'square_edge_by_perimeter': ('Div', 'temp_a', 4),
    'negate_prob': ('Sub', 1, 'temp_a'),
}

constants = {
    "const_pi": 3.1416,
    "const_5": 5,
    "const_2": 2,
    "const_2.0": 2,
    "const_1": 1,
    "const_3": 3,
    "const_3.0": 3,
    "const_4": 4,
    "const_4.0": 4,
    "const_6": 6,
    "const_12": 12,
    "const_10": 10,
    "const_100": 100,
    "const_100.0": 100,
    "const_1000": 1000,
    "const_26": 26,
    "const_52": 52,
    "const_60": 60,
    "const_60.0": 60,
    "const_360": 360,
    "const_3600": 3600,
    "const_1_6": 1.6,
    "const_0.6": 0.6,
    "const_0_6": 0.6,
    "const_0_2778": 0.2778,
    "const_0.3937": 0.3937,
    "const_0_3937": 0.3937,
    "const_2.54": 2.54,
    "const_0.4535": 0.4535,
    "const_2.2046": 2.2046,
    "const_3_6": 3.6,
    "const_deg_to_rad": 0.01745,
    "const_180": 180,
    "const_0.5": 0.5,
    "const_0.25": 0.25,
    "const_0_25": 0.25,
    "const_0_33": 0.33
}

import re
def parse_number(problem: str):
    res = re.finditer(r'[\d]+,?[\d]*\.?[\d]*', problem)
    # objs = []
    num_list = []
    spans = []
    for obj in res:
        assert obj.group(0) == problem[obj.span()[0]:obj.span()[1]]
        # objs.append((obj.group(0), problem[obj.span()[0]:obj.span()[1]], obj.span()))
        num_list.append(float(obj.group(0).replace(",", "")))
        spans.append(obj.span())
    # objs = [(obj.group(0), problem[obj.span()[0]:obj.span()[1]], obj.span()) for obj in res]
    return num_list, spans


def parse_answer(answer):
    candidates = []
    for item in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\./:]?\d*(?:[eE][-+]?\d+)?", answer):
        candidates.append(item)

    if not candidates:
        return 'none'

    if len(candidates) == 1:
        obj = candidates[0]
        if isinstance(obj, str) and obj.startswith('0') and not obj.startswith('0.'):
            obj = f"0.{obj}"
        return obj
    elif len(candidates) == 2:
        if '/' in answer or ':' in answer:
            return f'{candidates[0]} / {candidates[1]}'
        else:
            return candidates[0]
    else:
        return f'{candidates[0]} + {candidates[1]} / {candidates[2]}'

if __name__ == '__main__':
    # print(parse_number("how many integers between 362,855 and 852,755 have tens digit 1 and units digit 3 ? and 3 and 4.5"))
    option_string = "a ) a ) 17.33 , b ) b ) 2 , c ) c ) 18 , d ) d ) 16 , e ) e ) 13.21"
    options = [item for item in re.findall('[a-e] \) ([^,]*)', option_string)]
    print(options)