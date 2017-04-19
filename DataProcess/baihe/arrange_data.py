# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: arrange_data.py
@time: 2017/1/4 13:32
@contact: ustb_liubo@qq.com
@annotation: arrange_data
"""
import sys
import logging
from logging.config import fileConfig
import os
import pdb
import shutil

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def arrange_pic():
    folder = '/data/liubo/face/baihe/tmp'
    dst_folder = '/data/liubo/face/baihe/person6w'
    person_dic = {}
    for root_folder, sub_folder_list, sub_file_list in os.walk(folder):
        if len(sub_file_list) > 0:
            person_id = os.path.split(root_folder)[1]
            person_dic[person_id] = 1
            dst_person_folder = os.path.join(dst_folder, person_id)
            if os.path.exists(dst_person_folder):
                print 'has exist %s' %person_id
                continue
            else:
                shutil.copytree(root_folder, dst_person_folder)
                print len(person_dic)


if __name__ == '__main__':
    arrange_pic()
    pass
