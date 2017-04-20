# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: replace_folder.py
@time: 2016/8/29 11:31
@contact: ustb_liubo@qq.com
@annotation: replace_folder
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    pair_file = '/data/liubo/face/self_all_pair.txt'
    new_pair_file = '/data/liubo/face/self_all_pair_0.5_0.37.txt'
    f = open(new_pair_file, 'w')
    for line in open(pair_file):
        line = line.rstrip().replace('tmp', 'self_0.5_0.37')
        f.write(line+'\n')
    f.close()
