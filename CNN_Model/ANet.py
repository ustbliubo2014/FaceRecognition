# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: ANet.py
@time: 2016/9/30 12:48
@contact: ustb_liubo@qq.com
@annotation: ANet : 前几层CNN使用权值共享, 后面几层使用LocalConnectLayer
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

import numpy.random as random

from keras.models import Sequential
import keras.layers as layers
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.local import LocallyConnected2D
from keras.utils import np_utils



def build_ANet(input_shape, nb_class):
    # input_shape : (150, 150, 3)
    model = Sequential()

    model.add(Convolution2D(32, 11, 11, border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, 9, 9, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(LocallyConnected2D(16, 9, 9, border_mode='valid'))
    model.add(Activation('relu'))
    #
    # model.add(LocallyConnected2D(16, 7, 7, border_mode='valid'))
    # model.add(Activation('relu'))
    #
    # model.add(LocallyConnected2D(16, 5, 5, border_mode='valid'))
    # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    model = build_ANet(input_shape=(3, 150, 150), nb_class=1122)
    opt = SGD(momentum=0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(random.rand(32, 3, 150, 150), random.rand(32, 1122), batch_size=4, nb_epoch=5)
