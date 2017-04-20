#-*- coding: utf-8 -*-
__author__ = 'liubo-it'

from time import time

# def time_decorator(func):
#     def _deco(*args):
#         start = time()
#         func(*args)
#         end = time()
#         print('func.name', func.__name__, 'run_time',(end-start))
#     return _deco

# 函数可以带参数, 也可以有返回值
def time_decorator(func):
    def _deco(*args, **kwargs):
        start = time()
        ret = func(*args, **kwargs)
        end = time()
        print('func.name', func.__name__, 'run_time',(end-start))
        return ret
    return _deco