# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: center_loss.py
@time: 2016/11/1 18:24
@contact: ustb_liubo@qq.com
@annotation: center_loss
"""
import sys
import logging
from logging.config import fileConfig
import os
from keras.layers import Input
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.layers import merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

'''
    网络结构
        xavier/gaussian/constant 是初始化的方法, 卷积层一般用xavier/gaussian, bias用constant
        input : (112, 96, 3)
        conv1a : 32, 3, 3, PReLU
        conv1b : 64, 3, 3, xavier, PReLU
        pool1b : max
        conv2_1 : 64, 3, 3, gaussian, PReLU
        conv2_2 : 64, 3. 3, gaussian, PReLU
        pool2 : max
        conv3_1 : 128, 3, 3, gaussian, PReLU
        conv3_2 : 128, 3, 3, gaussian, PReLU
        conv3_3 : 128, 3, 3, gaussian, PReLU
        conv3_4 : 128, 3, 3, gaussian, PReLU
        pool3 : max
        conv4_1 : 256, 3, 3, gaussian, PReLU
        conv4_2 : 256, 3, 3, gaussian, PReLU
        conv4_3 : 256, 3, 3, gaussian, PReLU
        conv4_4 : 256, 3, 3, gaussian, PReLU
        conv4_5 : 256, 3, 3, gaussian, PReLU
        conv4_6 : 256, 3, 3, gaussian, PReLU
        conv4_7 : 256, 3, 3, gaussian, PReLU
        conv4_8 : 256, 3, 3, gaussian, PReLU
        conv4_9 : 256, 3, 3, gaussian, PReLU
        conv4_10 : 256, 3, 3, gaussian, PReLU
        pool4 : max
        conv5_1 : 512, 3, 3, gaussian, PReLU
        conv5_2 : 512, 3, 3, gaussian, PReLU
        conv5_3 : 512, 3, 3, gaussian, PReLU
        conv5_4 : 512, 3, 3, gaussian, PReLU
        conv5_5 : 512, 3, 3, gaussian, PReLU
        conv5_6 : 512, 3, 3, gaussian, PReLU
        fc5 : 512
'''

def create_model(input, nb_output):
    conv1a = Convolution2D(32, 3, 3, border_mode='same', name='conv1a')(input)
    conv1a = PReLU()(conv1a)
    conv1b = Convolution2D(64, 3, 3, border_mode='same', name='conv1b')(conv1a)
    conv1b = BatchNormalization(axis=1)(conv1b)
    conv1b = PReLU()(conv1b)
    pool1b = MaxPooling2D((2, 2), name='pool1b')(conv1b)

    conv2_1 = Convolution2D(64, 3, 3, border_mode='same', name='conv2_1')(pool1b)
    conv2_1 = PReLU()(conv2_1)
    conv2_2 = Convolution2D(64, 3, 3, border_mode='same', name='conv2_2')(conv2_1)
    conv2_2 = PReLU()(conv2_2)
    res2_2 = merge([pool1b, conv2_2], mode='sum', concat_axis=1)
    res2_2 = BatchNormalization(axis=1)(res2_2)
    conv2 = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(res2_2)
    pool2 = MaxPooling2D((2, 2), name='pool2')(conv2)

    conv3_1 = Convolution2D(128, 3, 3, border_mode='same', name='conv3_1')(pool2)
    conv3_1 = PReLU()(conv3_1)
    conv3_2 = Convolution2D(128, 3, 3, border_mode='same', name='conv3_2')(conv3_1)
    conv3_2 = PReLU()(conv3_2)
    res3_2 = merge([pool2, conv3_2], mode='sum', concat_axis=1)
    res3_2 = BatchNormalization(axis=1)(res3_2)

    conv3_3 = Convolution2D(128, 3, 3, border_mode='same', name='conv3_3')(res3_2)
    conv3_3 = PReLU()(conv3_3)
    conv3_4 = Convolution2D(128, 3, 3, border_mode='same', name='conv3_4')(conv3_3)
    conv3_4 = PReLU()(conv3_4)
    res3_4 = merge([res3_2, conv3_4], mode='sum', concat_axis=1)
    res3_4 = BatchNormalization(axis=1)(res3_4)

    conv3 = Convolution2D(256, 3, 3, border_mode='same', name='conv3')(res3_4)
    pool3 = MaxPooling2D((2, 2), name='pool3')(conv3)

    conv4_1 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_1')(pool3)
    conv4_1 = PReLU()(conv4_1)
    conv4_2 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_2')(conv4_1)
    conv4_2 = PReLU()(conv4_2)
    res4_2 = merge([pool3, conv4_2], mode='sum', concat_axis=1)
    res4_2 = BatchNormalization(axis=1)(res4_2)

    conv4_3 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_3')(res4_2)
    conv4_3 = PReLU()(conv4_3)
    conv4_4 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_4')(conv4_3)
    conv4_4 = PReLU()(conv4_4)
    res4_4 = merge([res4_2, conv4_4], mode='sum', concat_axis=1)
    res4_4 = BatchNormalization(axis=1)(res4_4)

    conv4_5 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_5')(res4_4)
    conv4_5 = PReLU()(conv4_5)
    conv4_6 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_6')(conv4_5)
    conv4_6 = PReLU()(conv4_6)
    res4_6 = merge([res4_4, conv4_6], mode='sum', concat_axis=1)
    res4_6 = BatchNormalization(axis=1)(res4_6)

    conv4_7 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_7')(res4_6)
    conv4_7 = PReLU()(conv4_7)
    conv4_8 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_8')(conv4_7)
    conv4_8 = PReLU()(conv4_8)
    res4_8 = merge([res4_6, conv4_8], mode='sum', concat_axis=1)
    res4_8 = BatchNormalization(axis=1)(res4_8)

    conv4_9 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_9')(res4_8)
    conv4_9 = PReLU()(conv4_9)
    conv4_10 = Convolution2D(256, 3, 3, border_mode='same', name='conv4_10')(conv4_9)
    conv4_10 = PReLU()(conv4_10)
    res4_10 = merge([res4_8, conv4_10], mode='sum', concat_axis=1)
    res4_10 = BatchNormalization(axis=1)(res4_10)

    conv4 = Convolution2D(512, 3, 3, border_mode='same', name='conv4')(res4_10)
    pool4 = MaxPooling2D((2, 2), name='pool4')(conv4)

    conv5_1 = Convolution2D(512, 3, 3, border_mode='same', name='conv5_1')(pool4)
    conv5_1 = PReLU()(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, border_mode='same', name='conv5_2')(conv5_1)
    conv5_2 = PReLU()(conv5_2)
    res5_2 = merge([pool4, conv5_2], mode='sum', concat_axis=1)
    res5_2 = BatchNormalization(axis=1)(res5_2)

    conv5_3 = Convolution2D(512, 3, 3, border_mode='same', name='conv5_3')(res5_2)
    conv5_3 = PReLU()(conv5_3)
    conv5_4 = Convolution2D(512, 3, 3, border_mode='same', name='conv5_4')(conv5_3)
    conv5_4 = PReLU()(conv5_4)
    res5_4 = merge([res5_2, conv5_4], mode='sum', concat_axis=1)
    res5_4 = BatchNormalization(axis=1)(res5_4)

    conv5_5 = Convolution2D(512, 3, 3, border_mode='same', name='conv5_5')(res5_4)
    conv5_5 = PReLU()(conv5_5)
    conv5_6 = Convolution2D(512, 3, 3, border_mode='same', name='conv5_6')(conv5_5)
    conv5_6 = PReLU()(conv5_6)
    res5_6 = merge([res5_4, conv5_6], mode='sum', concat_axis=1)
    res5_6 = BatchNormalization(axis=1)(res5_6)

    res5_6 = Flatten()(res5_6)
    fc5 = Dense(output_dim=512)(res5_6)
    fc5 = Dropout(0.5)(fc5)
    network = Dense(output_dim=nb_output, activation='softmax')(fc5)
    return network


if __name__ == '__main__':

    nb_classes = 10575
    model_file = '/data/liubo/face/annotate_face_model/skyeye_crop240_center_loss_%d.model' % (nb_classes)
    weight_file = '/data/liubo/face/annotate_face_model/skyeye_crop240_center_loss_%d.weight' % (nb_classes)

    ip = Input(shape=(3, 128, 128))

    inception_v4 = create_model(ip, nb_classes)
    model = Model(input=ip, output=inception_v4)

    open(model_file, 'w').write(model.to_json())
    model.save_weights(weight_file, overwrite=True)

    model.summary()
    plot(model, to_file="skyeye_center_loss_keras.png", show_shapes=True)
