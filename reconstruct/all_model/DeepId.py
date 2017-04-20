# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId.py
@time: 2016/7/27 15:39
@contact: ustb_liubo@qq.com
@annotation: DeepId
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
import pdb
from keras.models import (
    Sequential,
    Model,
    model_from_json
)
from keras.layers import (
    Dense,
    Dropout,
    Input,
    Lambda,
    merge,
    Convolution2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization
)
from keras.optimizers import (
    SGD,
    RMSprop,
    Adagrad,
    Adam
)
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import os
from time import time
import theano
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
from keras import backend as K
from model_conf import *
from util_model import conv2D_bn

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def build_deepId_conv_network(input_shape):
    print 'input_shape :', input_shape
    # (3, 128, 128)
    input_layer = Input(shape=input_shape)
    layer1 = conv2D_bn(input_layer, nb_filter=32)
    layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)

    layer2 = conv2D_bn(layer1, nb_filter=64)
    layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)

    layer3 = conv2D_bn(layer2, nb_filter=128)
    layer3 = MaxPooling2D(pool_size=(2, 2))(layer3)

    layer4 = conv2D_bn(layer3, nb_filter=128)
    layer4 = MaxPooling2D(pool_size=(2, 2))(layer4)

    layer5 = conv2D_bn(layer4, nb_filter=256)

    layer4_flatten = Flatten()(layer4)
    layer5_flatten = Flatten()(layer5)
    flatten_layer = merge(inputs=[layer4_flatten, layer5_flatten],
                          mode='concat')
    # 在softmax前增加一个隐含层,方便降维
    dense = Dense(hidden_num, activation='relu')(flatten_layer)

    model = Model(input=[input_layer], output=[dense])
    return model


def get_model(args):
    print('building deepid model')
    img_row = args.img_row
    img_col = args.img_col
    img_channel = args.img_channel
    nb_classes = args.nb_classes
    input_shape = (img_channel, img_row, img_col)
    base_network = build_deepId_conv_network(input_shape)
    input_layer = Input(shape=input_shape)
    conv_model = base_network(input_layer)
    pred_layer = Dense(nb_classes, activation='softmax')(conv_model)
    model = Model(input=[input_layer], output=[pred_layer])
    opt = RMSprop()
    # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    return model


# 模型的训练部分放在一个公共文件中,
