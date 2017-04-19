# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: lighted_cnn_a.py
@time: 2016/11/2 18:47
@contact: ustb_liubo@qq.com
@annotation: lighted_cnn
"""
import sys
import logging
from logging.config import fileConfig
import os
from keras.layers import Input
from keras.models import Model
from keras.layers import merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import keras.backend as K

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')



def create_model(input, nb_output):
    conv1a = Convolution2D(48, 5, 5, border_mode='same', name='conv1a')(input)
    conv1a = PReLU()(conv1a)
    conv1b = Convolution2D(48, 5, 5, border_mode='same', name='conv1b')(conv1a)
    conv1b = PReLU()(conv1b)
    conv1b = BatchNormalization()(conv1b)
    pool1b = MaxPooling2D((2, 2), name='pool1b')(conv1b)

    conv2_1 = Convolution2D(48, 1, 1, border_mode='same', name='conv2_1')(pool1b)
    conv2_1 = PReLU()(conv2_1)
    conv2_2 = Convolution2D(96, 3, 3, border_mode='same', name='conv2_2')(conv2_1)
    conv2_2 = PReLU()(conv2_2)
    conv2_3 = Convolution2D(96, 3, 3, border_mode='same', name='conv2_3')(conv2_2)
    conv2_3 = PReLU()(conv2_3)
    conv2_3 = BatchNormalization()(conv2_3)
    pool2 = MaxPooling2D((2, 2), name='pool2')(conv2_3)

    conv3_1 = Convolution2D(96, 1, 1, border_mode='same', name='conv3_1')(pool2)
    conv3_1 = PReLU()(conv3_1)
    conv3_2 = Convolution2D(192, 3, 3, border_mode='same', name='conv3_2')(conv3_1)
    conv3_2 = PReLU()(conv3_2)
    conv3_3 = Convolution2D(192, 3, 3, border_mode='same', name='conv3_3')(conv3_2)
    conv3_3 = PReLU()(conv3_3)
    conv3_3 = BatchNormalization()(conv3_3)
    pool3 = MaxPooling2D((2, 2), name='pool3')(conv3_3)

    conv4_1 = Convolution2D(192, 1, 1, border_mode='same', name='conv4_1')(pool3)
    conv4_1 = PReLU()(conv4_1)
    conv4_2 = Convolution2D(128, 3, 3, border_mode='same', name='conv4_2')(conv4_1)
    conv4_2 = PReLU()(conv4_2)
    conv4_3 = Convolution2D(128, 3, 3, border_mode='same', name='conv4_3')(conv4_2)
    conv4_3 = PReLU()(conv4_3)
    conv4_3 = BatchNormalization()(conv4_3)
    pool4 = MaxPooling2D((2, 2), name='pool4')(conv4_3)

    conv5_1 = Convolution2D(128, 1, 1, border_mode='same', name='conv5_1')(pool4)
    conv5_1 = PReLU()(conv5_1)
    conv5_2 = Convolution2D(128, 3, 3, border_mode='same', name='conv5_2')(conv5_1)
    conv5_2 = PReLU()(conv5_2)
    conv5_3 = Convolution2D(128, 3, 3, border_mode='same', name='conv5_3')(conv5_2)
    conv5_3 = PReLU()(conv5_3)
    conv5_3 = BatchNormalization()(conv5_3)

    faltten5 = Flatten()(conv5_3)
    fc5 = Dense(output_dim=256)(faltten5)
    fc5 = Dropout(0.5)(fc5)
    network = Dense(output_dim=nb_output, activation='softmax')(fc5)
    return network


if __name__ == '__main__':
    nb_classes = 558
    model_file = '/data/liubo/face/annotate_face_model/baihe_30_dlib_light_cnn_%d.model' % (nb_classes)
    weight_file = '/data/liubo/face/annotate_face_model/baihe_30_dlib_light_cnn_%d.weight' % (nb_classes)

    pic_size = 96

    if K.image_dim_ordering() == 'th':
        ip = Input(shape=(3, pic_size, pic_size))
    else:
        ip = Input(shape=(pic_size, pic_size, 3))

    light_cnn = create_model(ip, nb_classes)
    model = Model(input=ip, output=light_cnn)

    open(model_file, 'w').write(model.to_json())
    model.save_weights(weight_file, overwrite=True)

    model.summary()

