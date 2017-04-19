# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: baidu_face_stat.py
@time: 2016/7/30 20:54
"""

import logging
import os
import sys
import pdb
from conf import *


def get_sim(sim_str):
    return float(sim_str[-3:-1])


def find_right_url(url_file_name='url_check_result.txt'):
    # 找到已经解析的url
    right_url_dic = {}     # {person:[(index, url)]}
    for line in open(url_file_name):
        tmp = line.rstrip().split('\t')
        if len(tmp) >= 5:
            if tmp[3] != '' and timeout_str not in tmp[3] and analyse_error_str not in tmp[3]:
                person = tmp[0]
                this_person_list = right_url_dic.get(person, [])
                this_person_list.append((tmp[1], tmp[2]))
                right_url_dic[person] = this_person_list
    return right_url_dic


if __name__ == '__main__':
    find_right_url()