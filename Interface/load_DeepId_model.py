#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: load_deeid_model.py
@time: 2016/5/26 10:55
@annotation: 加载模型(一直在内存)
"""

import numpy as np
from keras.models import model_from_json
from keras import backend as K
import os

def load_deepid_model(model_file, weight_file):
    print 'model_file :', model_file
    print 'weight_file :', weight_file
    model = model_from_json(open(model_file, 'r').read())
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
    get_Conv_FeatureMap = K.function([model.layers[1].layers[0].get_input_at(False), K.learning_phase()],
                                     [model.layers[1].layers[-1].get_output_at(False)])
    # get_Conv_FeatureMap = theano.function([model.layers[1].layers[0].get_input_at(False)],
    #                                   model.layers[1].layers[-1].get_output_at(False))
    return model, get_Conv_FeatureMap


def extract_feature(pic_data, get_Conv_FeatureMap, feature_dim=98304):
    '''
    :param data: 需要提取特征的数据
    :param get_Conv_FeatureMap: 特征提取函数
    :param feature_dim: 特征维数
    :return:
    '''
    batch_size = 1
    # 测试阶段是一张图片提取一次
    batch_num = pic_data.shape[0] / batch_size
    pic_data_feature = np.zeros(shape=(pic_data.shape[0], feature_dim))
    for num in range(batch_num):
        batch_data = pic_data[num*batch_size:(num+1)*batch_size, :, :, :]
        pic_data_feature[num*batch_size:(num+1)*batch_size, :] = get_Conv_FeatureMap(batch_data[:, :])
    return pic_data_feature


if __name__ == '__main__':
    pass
