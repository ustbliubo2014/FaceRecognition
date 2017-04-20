# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: __init__.py.py
@time: 2016/7/4 15:18
@contact: ustb_liubo@qq.com
@annotation: __init__.py
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='__init__.py.log',
                    filemode='w')

if __name__ == '__main__':
    pass
