# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: cal_distance.py
@time: 2016/7/4 17:18
@contact: ustb_liubo@qq.com
@annotation: cal_distance:计算一个文件夹下所有文件的相似度
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='cal_distance.log',
                    filemode='w')

if __name__ == '__main__':
    pass
