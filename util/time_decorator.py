#-*- coding: utf-8 -*-
__author__ = 'liubo-it'

from time import time

def time_decorator(func):
    def _deco(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print('func.name',func.__name__, 'run_time',(end-start))
    return _deco


@time_decorator
def foo2():
    return [3 + 5]

foo2()