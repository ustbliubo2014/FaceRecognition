# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: ModelFusion.py
@time: 2016/8/13 23:27
"""

import logging
import os
import sys
import msgpack_numpy
import pdb
import numpy as np


def model_fusion(feature_file1, feature_file2, feature_file3, feature_file):
    (all_data1, all_label1, all_pic_path_list1) = msgpack_numpy.load(open(feature_file1, 'rb'))
    (all_data2, all_label2, all_pic_path_list2) = msgpack_numpy.load(open(feature_file2, 'rb'))
    (all_data3, all_label3, all_pic_path_list3) = msgpack_numpy.load(open(feature_file3, 'rb'))
    all_data = []
    all_label = []
    all_pic_path_list = []

    length = len(all_data1)
    for index in range(length):
        if all_label1[index] != all_label2[index] or all_label1[index] != all_label3[index]:
            pdb.set_trace()
        # all_data.append(np.column_stack((all_data1[0], all_data2[0], all_data3[0])))
        all_data.append(all_data3[index])
        all_label.append(all_label1[index])
        all_pic_path_list.append(all_pic_path_list1[index])
    msgpack_numpy.dump((all_data, all_label, all_pic_path_list), open(feature_file, 'wb'))



if __name__ == '__main__':
    feature_file1 = '/data/liubo/face/annotate_self_feature_fc7.p'
    feature_file2 = '/data/liubo/face/annotate_self_feature_fc8.p'
    feature_file3 = '/data/liubo/face/annotate_self_feature_fc9.p'
    feature_file = '/data/liubo/face/annotate_self_feature.p'
    model_fusion(feature_file1, feature_file2, feature_file3, feature_file)
