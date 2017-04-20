#-*- coding:utf-8 -*-
__author__ = 'liubo-it'


import os
from random import randint
from scipy.misc import imread
import pdb

if __name__ == '__main__':
    folder = '/home/data/dataset/images/lfw-deepfunneled'
    one_count = 0
    multi_count = 0
    all_shape = set()
    file_list = os.listdir(folder)
    multi_count_dic = {}
    for dir_path in file_list:
        dir_path = os.path.join(folder, dir_path)
        png_list = os.listdir(dir_path)
        if len(png_list) == 1:
            one_count += 1
        else:
            for png in png_list:
                all_shape.add(imread(os.path.join(dir_path,png)).shape)
            multi_count += 1
            multi_count_dic[len(png_list)] = multi_count_dic.get(len(png_list),0) + 1
    pdb.set_trace()
    print one_count, multi_count, all_shape

