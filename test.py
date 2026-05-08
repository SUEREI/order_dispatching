def foo(d):
    d['new'] = 'value'     # 对 d 指向的对象做原地修改
    d['list'].append(123)  # 同上

my_dict = {'old': 'value', 'list': []}
foo(my_dict)
print(my_dict)  # {'old': 'value', 'list': [123], 'new': 'value'} — 改了！