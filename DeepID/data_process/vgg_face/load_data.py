#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: load_data.py
@time: 2016/5/31 10:51
@contact: ustb_liubo@qq.com
@annotation: load_data
"""

import os
import numpy as np
from scipy.misc import imread, imresize
import traceback
import pdb

def load_test_data(test_folder, person_num_threshold=100):
    person_list = os.listdir(test_folder)
    all_data = []
    all_label = []
    label_trans_dic = {}
    current_label = 0
    pic_shape = (128, 128, 3)
    for person in person_list:
        try:
            label_trans_dic[current_label] = person
            pic_folder = os.path.join(test_folder, person)
            pic_list = os.listdir(pic_folder)
            pic_list.sort()
            this_person_num = 0
            for pic in pic_list:
                absolute_path = os.path.join(pic_folder, pic)
                pic_arr = imresize(imread(absolute_path), pic_shape)
                if pic_arr.shape != pic_shape:
                    continue
                all_data.append(pic_arr)
                all_label.append(current_label)
                this_person_num += 1
                if this_person_num > person_num_threshold:
                    # 每个样本最多取100张图片
                    break
            current_label += 1
        except:
            traceback.print_exc()
            continue
    all_data = np.asarray(all_data, dtype=np.float32)
    all_data = np.dot(all_data[...,:3], [0.299, 0.587, 0.144])
    all_label = np.asarray(all_label)
    all_data = all_data / 255.0
    return all_data, all_label, label_trans_dic

def load_gray_test_data(test_folder, person_num_threshold=100):
    person_list = os.listdir(test_folder)
    all_data = []
    all_label = []
    label_trans_dic = {}
    current_label = 0
    pic_shape = (128, 128)
    for person in person_list:
        try:
            label_trans_dic[current_label] = person
            pic_folder = os.path.join(test_folder, person)
            pic_list = os.listdir(pic_folder)
            pic_list.sort()
            this_person_num = 0
            for pic in pic_list:
                absolute_path = os.path.join(pic_folder, pic)
                pic_arr = imread(absolute_path)
                pic_arr = imresize(pic_arr, pic_shape)
                if pic_arr.shape != pic_shape:
                    continue
                all_data.append(pic_arr)
                all_label.append(current_label)
                this_person_num += 1
                if this_person_num > person_num_threshold:
                    # 每个样本最多取100张图片
                    break
            current_label += 1
        except:
            traceback.print_exc()
            continue
    all_data = np.asarray(all_data, dtype=np.float32)
    all_label = np.asarray(all_label)
    all_data = all_data / 255.0
    return all_data, all_label, label_trans_dic


if __name__ == '__main__':
    test_folder = '/data/liubo/face/self'
    all_data, all_label, label_trans_dic = load_gray_test_data(test_folder)
    print all_label.shape, all_data.shape
