# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId.py
@time: 2016/7/27 15:39
@contact: ustb_liubo@qq.com
@annotation: DeepId
"""
from base_args import get_basic_args
import sys
import logging
from logging.config import fileConfig
reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


# 在该文件中只是用文件名

def get_args():
    person_num = 500
    img_row = 39
    img_col = 31
    img_channel = 3
    data_num = 40000
    nb_classes = 554
    pack_file = '/data/annotate_list.p'
    func_args_dic = {}
    args = get_basic_args(person_num, img_row, img_col, img_channel, data_num, pack_file, func_args_dic, nb_classes)
    return args

if __name__ == '__main__':
    args = get_args()
    print args.weight_file
    print args.model_file
