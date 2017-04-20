# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: deepface.py
@time: 2016/8/8 11:52
@contact: ustb_liubo@qq.com
@annotation: deepface
"""
import sys
import logging
from logging.config import fileConfig
import os
from keras.models import Model
from keras.layers import (
    Input,
    Convolution2D,
    MaxPooling2D,
    merge,
    Dense,
    BatchNormalization,
    Dropout
)
from keras.layers.core import Flatten
from keras.optimizers import (
    SGD,
    RMSprop,
    Adagrad,
    Adam
)
import msgpack
from time import time
import traceback
from keras import regularizers

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

def deep_face(input_shape, nb_classes):
    print 'input_shape :', input_shape
    input_layer = Input(shape=input_shape, name='input')
    layer1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv1_1')(input_layer)
    layer1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv1_2')(layer1)
    layer1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(layer1)

    layer2 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv2_1')(layer1)
    layer2 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv2_2')(layer2)
    layer2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(layer2)

    layer3 = Convolution2D(nb_filter=256, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv3_1')(layer2)
    layer3 = Convolution2D(nb_filter=256, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv3_2')(layer3)
    layer3 = Convolution2D(nb_filter=256, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv3_3')(layer3)
    layer3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(layer3)

    layer4 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv4_1')(layer3)
    layer4 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv4_2')(layer4)
    layer4 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv4_3')(layer4)
    layer4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(layer4)

    layer5 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv5_1')(layer4)
    layer5 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv5_2')(layer5)
    layer5 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv5_3')(layer5)
    layer5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5')(layer5)

    flatten_layer = Flatten(name='flatten')(layer5)
    layer6 = Dense(4096, activation='relu', name='fc6')(flatten_layer)
    layer6 = Dropout(0.5)(layer6)
    layer7 = Dense(4096, activation='relu', name='fc7')(layer6)
    layer7 = Dropout(0.5)(layer7)

    layer8 = Dense(nb_classes, activation='softmax', name='prob')(layer7)

    model = Model(input=[input_layer], output=[layer8])
    print model.summary()

    opt = RMSprop()
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    return model


if __name__ == '__main__':
    print deep_face(input_shape=(3, 128, 128), nb_classes=2622)
