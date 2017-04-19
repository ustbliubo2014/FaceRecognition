# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: Data.py
@time: 2016/7/15 10:46
@contact: ustb_liubo@qq.com
@annotation: Data
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='Data.log',
                    filemode='a+')

class ImageData():
    def __init__(self, init_args):
        pass

if __name__ == '__main__':
    pass
