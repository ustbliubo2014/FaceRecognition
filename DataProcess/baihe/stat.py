# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: stat.py
@time: 2017/1/5 16:37
@contact: ustb_liubo@qq.com
@annotation: stat
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def stat_pic_num(folder):
    pic_num = 0
    person_num = 0
    pic_num_dic = {}
    all_person_num = 0
    for root_folder, sub_folder_list, sub_file_list in os.walk(folder):
        if all_person_num % 5000 == 0:
            print all_person_num
        this_pic_num = len(sub_file_list)
        if this_pic_num == 0 and len(sub_folder_list) == 0:
            all_person_num += 1
            continue
        elif this_pic_num > 0 and len(sub_folder_list) == 0:
            # if this_pic_num > 50:
            #     print root_folder
            all_person_num += 1
            person_num += 1
            pic_num += this_pic_num
            pic_num_dic[this_pic_num] = pic_num_dic.get(this_pic_num, 0) + 1
    print 'person_num :', person_num, 'pic_num :', pic_num
    print pic_num_dic


if __name__ == '__main__':
    # stat_pic_num(folder='/data/liubo/face/baihe/person_mtcnn_160')
    # stat_pic_num(folder='/data/liubo/face/MS-Celeb_face/clean_data_face_aling')
    stat_pic_num(folder='/data/liubo/face/baihe/verif/person_mtcnn_128')
    pass
