# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: filter_data.py
@time: 2016/9/5 15:17
@contact: ustb_liubo@qq.com
@annotation: filter_data
"""
import sys
import logging
from logging.config import fileConfig
import os
import pdb
from scipy.misc import imread
import shutil

reload(sys)
sys.setdefaultencoding("utf-8")

min_shape = 96
person_pic = 20

if __name__ == '__main__':
    folder = '/data/MS-Celeb_face'
    new_folder = '/data/MS-Celeb_face_filter'
    person_list = os.listdir(folder)
    all_count = 0
    all_del_count = 0
    person_num = 0
    for person_index, person in enumerate(person_list):
        mean_width = mean_height = 0
        person_path = os.path.join(folder, person)
        pic_list = map(lambda x:os.path.join(person_path, x), os.listdir(person_path))
        for pic in pic_list:
            arr = imread(pic)
            mean_height += arr.shape[1]
            mean_width += arr.shape[0]
            if len(arr.shape) == 3 and arr.shape[0] > min_shape and arr.shape[1] > min_shape and arr.shape[2] == 3:
                continue
            else:
                all_del_count += 1
                os.remove(pic)
        all_count += len(pic_list)
        if len(os.listdir(person_path)) > person_pic:
            person_num += 1
            new_person_path = os.path.join(new_folder, person)
            shutil.copytree(person_path, new_person_path)
        if person_index % 100 == 0:
            print all_count, all_del_count, person_num
