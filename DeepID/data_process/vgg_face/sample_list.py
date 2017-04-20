#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: sample_list.py
@time: 2016/5/25 11:17
"""

import os
import msgpack
from time import time
import numpy as np
from scipy.misc import imread, imresize, imsave
import shutil

def create_sample_list(vgg_folder, sample_list_file, shape=(128, 128, 3)):
    '''
        :param vgg_folder: /data/liubo/face/vgg_face_dataset/all_data/pictures_box
        :param sample_list_file: [(pic_path,person_id)]
        :return:
    '''
    person_list = os.listdir(vgg_folder)
    person_id_trans_dic = {}#{person:id}
    current_person_id = 0
    sample_list = []
    for person in person_list:
        count = 0
        start = time()
        person_id = person_id_trans_dic.get(person, current_person_id)
        person_path = os.path.join(vgg_folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            arr = imread(pic_path)
            if arr.shape != shape:
                os.remove(pic_path)
                continue
            sample_list.append((pic_path,person_id))
            count += 1
        current_person_id += 1
        end = time()
        print person, count, (end-start)
    start = time()
    np.random.shuffle(sample_list)
    end = time()
    print 'shuffle time : ', (end-start), 'all sample : ', (len(sample_list))
    msgpack.dump(sample_list, open(sample_list_file, 'wb'))

def split_train_valid(sample_list, valid_person_num=50):
    # 每个人50张图片用于验证
    train_sample_list = []
    valid_sample_list = []
    valid_person_dic = {}#{person_id:valid_num}
    for pic_path, person_id in sample_list:
        if valid_person_dic.get(person_id, 0) < valid_person_num:
            valid_sample_list.append((pic_path, person_id))
            valid_person_dic[person_id] = valid_person_dic.get(person_id, 0) + 1
        else:
            train_sample_list.append((pic_path, person_id))
    return train_sample_list, valid_sample_list

if __name__ == '__main__':
    # vgg_folder = '/data/liubo/face/vgg_face_dataset/all_data/pictures_box'
    # sample_list_file = '/data/liubo/face/vgg_face_dataset/all_data/sample_list.p'
    # create_sample_list(vgg_folder, sample_list_file)
    # sample_list = msgpack.load(open(sample_list_file,'rb'))
    # train_sample_list, valid_sample_list = split_train_valid(sample_list)
    # msgpack.dump((train_sample_list, valid_sample_list), open(sample_list_file,'wb'))
    #

    sample_list_file = '/data/liubo/face/vgg_face_dataset/all_data/all_sample_list.p'
    valid_folder = '/data/liubo/face/vgg_face_dataset/all_data/valid'
    train_sample_list, valid_sample_list = msgpack.load(open(sample_list_file,'rb'))
    for valid_pic_path, person_id in valid_sample_list:
        tmp = valid_pic_path.split('/')
        person = tmp[-2]
        person_dir = os.path.join(valid_folder, person)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        shutil.copy(valid_pic_path, person_dir)

