# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: person_pic_arrangement.py
@time: 2016/8/1 15:51
@contact: ustb_liubo@qq.com
@annotation: person_pic_arrangement : 根据百度给出的两个标题决定哪些图片需要标注,哪些图片不需要标注
"""

import sys
import logging
from logging.config import fileConfig
import os
import msgpack
import pdb
import traceback
import shutil
from conf import *

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


no_meaning_list = ['', analyse_error1_str, timeout1_str]
no_find_baike = no_newbaike_name
sim_threshold = 0.8


def get_newbaike_sim(newbaike_sim):
    try:
        return float(newbaike_sim[-3:-1])
    except:
        traceback.print_exc()
        return 0.0


def load_check_result_url(dic_file, check_url_file):
    person_result_dic = {}   # {person:([](right_set),[](wrong_set))} # 肯定正确和肯定错的的图片
    right_url_count = wrong_url_count = error_format_count = no_baike_count = no_meaning_count = 0
    if os.path.exists(dic_file):
        person_result_dic = msgpack.load(open(dic_file, 'rb'))
    for line in open(check_url_file):
        tmp = line.rstrip().split('\t')
        # [person_name, pic_index, pic_url, baike_name, baike_sim, newbaike_sim, guess_info]
        person_name = tmp[0]
        right_list, wrong_list = person_result_dic.get(person_name, ([], []))
        if len(tmp) == 7:
            if tmp[3] not in no_meaning_list:
                if tmp[3] == no_find_baike:
                    no_baike_count += 1
                    continue
                else:
                    if get_newbaike_sim(tmp[4]) > sim_threshold:
                        if tmp[0] == tmp[3]:
                            right_list.append(tmp[1])
                            right_url_count += 1
                        else:
                            wrong_url_count += 1
                            wrong_list.append(tmp[1])
                    else:   # 小于某概率时结果不可信,需要标注
                        no_baike_count += 1
                        continue
            else:
                no_meaning_count += 1
                continue
        else:
            error_format_count += 1
            continue
        person_result_dic[person_name] = (right_list, wrong_list)
    print right_url_count, wrong_url_count, no_baike_count, no_meaning_count, error_format_count
    msgpack.dump(person_result_dic, open('person_result_dic.p', 'w'))


def move_pic():
    pic_folder = '/data/pictures_face/'
    right_pic_folder = '/data/pictures_face_baidu_filter/'
    need_annotate_folder = '/data/pictures_face_need_annotate/'
    person_result_dic = msgpack.load(open('person_result_dic.p', 'r'))
    person_list = os.listdir(pic_folder)
    for person in person_list:
        old_person_path = os.path.join(pic_folder, person)
        right_person_path = os.path.join(right_pic_folder, person)
        annotate_person_path = os.path.join(need_annotate_folder, person)
        right_index_list, wrong_index_list = person_result_dic.get(person.decode('gbk').encode('utf-8'), ([], []))
        right_index_list = set(right_index_list)
        wrong_index_list = set(wrong_index_list)
        old_pic_list = os.listdir(old_person_path)
        for pic in old_pic_list:
            pic_index = pic.replace('.png', '').replace('0.jpg', '').replace('_', '')
            if pic_index in right_index_list:
                if not os.path.exists(right_person_path):
                    os.makedirs(right_person_path)
                shutil.copyfile(os.path.join(old_person_path, pic),
                                os.path.join(right_person_path, pic))
            elif pic_index in wrong_index_list:
                continue
            else:
                if not os.path.exists(annotate_person_path):
                    os.makedirs(annotate_person_path)
                shutil.copyfile(os.path.join(old_person_path, pic),
                                os.path.join(annotate_person_path, pic))


if __name__ == '__main__':
    load_check_result_url(dic_file='person_result_dic.p', check_url_file='url_check_result.txt')
    # move_pic()
