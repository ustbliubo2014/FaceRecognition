# -*- coding:utf-8 -*-
__author__ = 'liubo-it'


import sys
import os
from scipy.misc import imread, imresize
import numpy as np
import msgpack_numpy


def train_valid_split(raw_data_folder, msgpack_data_file, youtube_size=(55, 47, 3), split_rate=0.7):
    '''
        lfw原始图像的大小 (250, 250, 3), 需要用imresize转换成youtube的大小(55, 47, 3)
    :param raw_data_folder: lfw的文件夹
    :return:
    '''
    name_folder_list = os.listdir(raw_data_folder)
    name_folder_list.sort()
    train_data = []
    train_label = []
    valid_data = []
    valid_label = []
    label_trans_dic = {}
    current_label = 0
    for name_folder in name_folder_list:
        png_file_list = os.listdir(os.path.join(raw_data_folder, name_folder))
        if len(png_file_list) < 2:
            continue
        length = len(png_file_list)
        for index, png_file in enumerate(png_file_list):
            path = os.path.join(raw_data_folder, name_folder, png_file)
            new_arr = imresize(imread(path), youtube_size)
            if index < length * split_rate:
                train_data.append(new_arr)
                train_label.append(current_label)
            else:
                valid_data.append(new_arr)
                valid_label.append(current_label)
        label_trans_dic[name_folder] = current_label
        current_label += 1
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    valid_data = np.asarray(valid_data)
    valid_label = np.asarray(valid_label)
    msgpack_numpy.dump((train_data,train_label,valid_data,valid_label),open(msgpack_data_file,'wb'))
    print train_data.shape, train_label.shape, valid_data.shape, valid_label.shape


if __name__ == '__main__':
    raw_data_folder = '/home/data/dataset/images/lfw-deepfunneled/'
    msgpack_data_file = '/home/data/dataset/images/lfw_data/train_valid_data.p'
    train_valid_split(raw_data_folder, msgpack_data_file, youtube_size=(55, 47, 3), split_rate=0.7)
