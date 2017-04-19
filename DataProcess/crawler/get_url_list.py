# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: get_url_list.py
@time: 2016/8/1 11:19
@contact: ustb_liubo@qq.com
@annotation: get_url_list
"""
import sys
import logging
from logging.config import fileConfig
import os
import msgpack
import traceback
from time import time

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def find_url_list():
    pic_face_index_dic = msgpack.load(open('pic_face_index_dic.p', 'rb'))
    url_folder = '/data/url'
    for person in pic_face_index_dic:
        print person
        url_list = open(os.path.join(url_folder, person+'.txt'),
                        'r').read().split('\n')
        need_check_url_index_list = pic_face_index_dic.get(person)
        for index in range(len(need_check_url_index_list)):
            tmp = url_list[int(need_check_url_index_list[index])].split('\t')
            need_check_url_index_list[index] = \
                (need_check_url_index_list[index], tmp[-1])
        pic_face_index_dic[person] = need_check_url_index_list
    msgpack.dump(pic_face_index_dic, open('pic_face_index_url_dic.p', 'wb'))


def split_all_url():
    '''
        将url列表以人分成多个文件,然后用hadoop爬数据
    '''
    result_folder = 'person_url_check'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    pic_face_index_dic = msgpack.load(open('pic_face_index_url_dic.p', 'rb'))
    person_count = 0
    url_count = 0
    person_index = 0
    for person in pic_face_index_dic:
        start = time()
        person_index += 1
        with open(os.path.join(result_folder, str(person_index)), 'w') as f_result:
            try:
                need_check_url_index_list = pic_face_index_dic.get(person)
                for index, pic_url in need_check_url_index_list:
                    write_content = [person, index, pic_url]
                    f_result.write('\t'.join(map(str, write_content))+'\n')
                    url_count += 1
            except:
                traceback.print_exc()
                continue
            person_count += 1
            print person, person_count, url_count, time()-start


def find_url_index(pic_face_folder):
    pic_face_index_dic = {}         # {person_name:[2,6,9]}
    person_list = os.listdir(pic_face_folder)
    for person_index, person in enumerate(person_list):
        print person_index, person.decode('gbk').encode('utf-8')
        person_path = os.path.join(pic_face_folder, person)
        pic_list = os.listdir(person_path)
        this_index_list = []
        for pic in pic_list:
            if 'face' not in pic:
                try:
                    index = pic.replace('.png', '')
                    this_index_list.append(index)
                except:
                    traceback.print_exc()
        person = person.decode('gbk').encode('utf-8')
        pic_face_index_dic[person] = this_index_list
    msgpack.dump(pic_face_index_dic, open('pic_face_index_dic.p', 'wb'))


if __name__ == '__main__':
    pass
