# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: pic_group.py
@time: 2016/7/27 21:29
"""

import logging
import os
import sys
import shutil

one_person_pic_threshold = 50

def filter(pic_face_folder):
    person_list = os.listdir(pic_face_folder)
    filter_num = 0
    for person in person_list:
        person_path = os.path.join(pic_face_folder, person)
        pic_list = os.listdir(person_path)
        if len(pic_list) < one_person_pic_threshold:
            print person
            filter_num += 1
            shutil.rmtree(person_path)
    print len(person_list), filter_num


def get_pic_path(face_path):
    return face_path.split('_')[0]


def move_pic(pic_folder, pic_face_folder):
    person_face_list = os.listdir(pic_face_folder)
    for person in person_face_list:
        this_person_face_path = os.path.join(pic_face_folder, person)
        face_list = os.listdir(this_person_face_path)
        for face in face_list:
            pic_path = get_pic_path(face)
            pic_real_path = os.path.join(pic_folder, person, pic_path)
            shutil.copy(pic_real_path, this_person_face_path)


def group_pic(pic_face_folder, group_folder, group_num=12):
    person_list = os.listdir(pic_face_folder)
    for group_index in range(group_num):
        this_group_folder = os.path.join(group_folder, str(group_index))
        if not os.path.exists(this_group_folder):
            os.makedirs(this_group_folder)
        for person_index in range(group_index, len(person_list), group_num):
            person_path = os.path.join(pic_face_folder, person_list[person_index])
            dst_path = os.path.join(this_group_folder, person_list[person_index])
            shutil.copytree(person_path, dst_path)
            print person_index


if __name__ == '__main__':
    pic_face_folder = '/data/pictures_face/'
    pic_folder = '/data/pictures/'
    group_folder = '/data/pictures_group/'
    # filter(pic_face_folder)
    move_pic(pic_folder, pic_face_folder)
    group_pic(pic_face_folder, group_folder)
