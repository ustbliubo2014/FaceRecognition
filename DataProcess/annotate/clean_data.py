# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: clean_data.py
@time: 2016/8/2 16:25
@contact: ustb_liubo@qq.com
@annotation: clean_data
"""
import sys
import logging
from logging.config import fileConfig
import os
sys.path.insert(0, '/home/liubo-it/FaceRecognization/')
from recog_util.load_data import load_two_deep_path
import shutil
import pdb
import numpy as np
import msgpack
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def get_face_index(file_name):
    return os.path.basename(file_name).split('.')[0]


def get_pic_index(file_name):
    return os.path.basename(file_name).split('.')[0]


def clean():
    person_path_dic = load_two_deep_path('/data/pictures_annotate')
    for person in person_path_dic:
        try:
            print person
            path_list = person_path_dic.get(person)
            face_index_list = []
            pic_index_list = []
            for path in path_list:
                # 86.png_face_0.jpg  86.png
                if 'face' in path:
                    face_index_list.append(get_face_index(path))
                else:
                    pic_index_list.append(get_pic_index(path))
            right_index = set(face_index_list) & set(pic_index_list)
            for path in path_list:
                if 'face' in path:
                    if get_face_index(path) not in right_index:
                        os.remove(path)
                else:
                    os.remove(path)
        except:
            traceback.print_exc()
            continue

def stat():
    person_path_dic = load_two_deep_path('/data/pictures_annotate')
    count = 0
    for person in person_path_dic:
        count += len(person_path_dic.get(person))
    train_list = []
    valid_list = []
    current_id = 0
    for person in person_path_dic:
        path_list = person_path_dic.get(person)
        train_num = int(len(path_list) * 0.9)
        valid_num = len(path_list) - train_num
        np.random.shuffle(path_list)
        train_list.extend(zip(path_list[:train_num], [current_id]*train_num))
        valid_list.extend(zip(path_list[valid_num:], [current_id]*valid_num))
        current_id += 1
    np.random.shuffle(train_list)
    np.random.shuffle(valid_list)
    print count, len(person_path_dic)
    msgpack.dump((train_list, valid_list), open('/data/annotate_list.p', 'wb'))


if __name__ == '__main__':
    clean()
    # stat()
