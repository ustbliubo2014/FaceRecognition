# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId.py
@time: 2016/7/27 15:39
@contact: ustb_liubo@qq.com
@annotation: util : 包含模型的一些基本操作
"""

from keras.models import model_from_json
import numpy as np
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
import importlib


def save_model(model, model_file):
    open(model_file, 'w').write(model.to_json())


def save_weight(model, weight_file):
    model.save_weights(weight_file, overwrite=True)


def load_model(model_file):
    return model_from_json(open(model_file).read())


def load_args(args_file):
    args = importlib.import_module(args_file).get_args()
    return args


def load_weight(model, weight_file):
    model.load_weights(weight_file)


def CRPS(label, pred):
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


def smallModel_2_bigModel(small_model, big_model):
    for index in range(len(small_model.layers)):
        if type(small_model.layers[index]) == Convolution2D or \
                        type(small_model.layers[index]) == BatchNormalization:
            big_model.layers[index].set_weights(
                small_model.layers[index].get_weights())
    return big_model


def log_print(logger, print_list):
    logger.error('\t'.join(map(str, print_list)))