# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: crawler_baidu_face_remove.py : 将已经爬到的数据保存
@time: 2016/7/30 21:19
"""

import logging
import os
import sys
from baidu_face_stat import find_right_url
import pdb
from conf import *

def load_all_url(folder='person_url_check'):

    dic = {}
    file_list = os.listdir(folder)
    for file in file_list:
        file_path = os.path.join(folder, file)
        this_person_list = []
        this_person = ''
        for line in open(file_path):
            tmp = line.rstrip().split()
            if len(tmp) == 3:
                this_person = tmp[0]
                this_person_list.append((tmp[1], tmp[2]))
        dic[this_person] = this_person_list
    return dic


def remove():
    all_url_dic = load_all_url(folder='person_url_check_2')
    right_url_dic = find_right_url(url_file_name='url_check_result_2.txt')
    new_folder = 'person_url_check_3'
    need_crawler_url_dic = {}    # {person: [(index,pic_url)]}
    for person in all_url_dic:
        this_person_all_url_list = all_url_dic.get(person)
        this_person_find_url_list = right_url_dic.get(person, [])
        this_person_need_crawler_url_list = []
        if len(this_person_find_url_list) > 0:
            this_index_url_dic = dict(this_person_find_url_list)
            for index, url in this_person_all_url_list:
                if index not in this_index_url_dic:
                    this_person_need_crawler_url_list.append((index, url))
            if len(this_person_need_crawler_url_list) > 0:
                need_crawler_url_dic[person] = this_person_need_crawler_url_list
        else:
            need_crawler_url_dic[person] = all_url_dic.get(person)

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    person_index = 0
    for person in need_crawler_url_dic:
        file_name = os.path.join(new_folder, str(person_index))
        print file_name
        need_check_url_list = need_crawler_url_dic.get(person)
        with open(file_name, 'w') as f:
            for index, url in need_check_url_list:
                f.write('\t'.join(map(str, [person, index, url]))+'\n')
            f.close()
            person_index += 1


if __name__ == '__main__':
    remove()