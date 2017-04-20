# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: filter.py
@time: 2016/7/4 17:14
@contact: ustb_liubo@qq.com
@annotation: filter : 同一个文件夹下可能存在多张相同的图片,根据阈值过滤图片
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='filter.log',
                    filemode='w')

if __name__ == '__main__':
    pass
