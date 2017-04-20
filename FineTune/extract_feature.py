# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: extract_feature.py
@time: 2016/8/17 19:09
@contact: ustb_liubo@qq.com
@annotation: extract_feature
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import msgpack_numpy
from util import load_model, read_one_rgb_pic
from load_data import load_orl, load_verif_originalimages
from conf import *
import pdb


avg = np.array([129.1863, 104.7624, 93.5940])


def extract(pic_path, get_Conv_FeatureMap, pic_shape):
    img = read_one_rgb_pic(pic_path, pic_shape, avg)
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    return feature_vector


def extract_orl_conv3():
    pic_shape = (128, 128, 3)
    model, get_Conv_FeatureMap = load_model(output_layer_index=10)
    path_list, label_list = load_orl()
    feature_list = []
    for index, pic_path in enumerate(path_list):
        feature_vector = extract(pic_path, get_Conv_FeatureMap, pic_shape)[0]
        feature_list.append(feature_vector)
    feature_list = np.asarray(feature_list)
    label_list = np.asarray(label_list)
    return feature_list, label_list


def extract_orl_fc7():
    pic_shape = (224, 224, 3)
    model, get_Conv_FeatureMap = load_model(output_layer_index=22)
    path_list, label_list = load_orl()
    feature_list = []
    for index, pic_path in enumerate(path_list):
        feature_vector = extract(pic_path, get_Conv_FeatureMap, pic_shape)[0]
        feature_list.append(feature_vector)
    feature_list = np.asarray(feature_list)
    label_list = np.asarray(label_list)
    return feature_list, label_list


def extract_verif_originalimages_fc7():
    pic_shape = (224, 224, 3)
    model, get_Conv_FeatureMap = load_model(output_layer_index=22)
    path_list, label_list = load_verif_originalimages()
    path_feature_dic = {}
    for index, pic_path in enumerate(path_list):
        feature_vector = extract(pic_path, get_Conv_FeatureMap, pic_shape)[0]
        path_feature_dic[pic_path] = feature_vector
    return path_feature_dic


def save_data(extract_func, data_path):
    feature, label = extract_func()
    msgpack_numpy.dump((feature, label), open(data_path, 'wb'))


def load_data(data_path):
    feature, label = msgpack_numpy.load(open(data_path, 'rb'))
    return feature, label


if __name__ == '__main__':
    # path_feature_dic = extract_verif_originalimages_fc7()
    # msgpack_numpy.dump(path_feature_dic, open(originalimages_verif_fc7_path_feature, 'wb'))

    save_data(extract_orl_fc7, orl_fc7_data_path)
    feature, label = load_data(orl_fc7_data_path)
    print feature.shape, label.shape