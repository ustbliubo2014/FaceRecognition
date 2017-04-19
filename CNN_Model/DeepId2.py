# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: DeepId2.py
@time: 2016/9/30 15:32
@contact: ustb_liubo@qq.com
@annotation: DeepId2
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
import pdb
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Input, Lambda, merge, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from DeepId import euclidean_distance, eucl_dist_output_shape, contrastive_loss, create_deepId_network

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def build_deepid2_model(input_shape, nb_classes):
    print('building deepid2 model')
    base_network = create_deepId_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    pred_a = Dense(nb_classes, activation='softmax')(processed_a)
    pred_b = Dense(nb_classes, activation='softmax')(processed_b)
    model = Model(input=[input_a, input_b], output=[distance, pred_a, pred_b])
    # model = Model(input=[input_a, input_b], output=[distance])
    opt = SGD(momentum=0.9)
    # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
    model.compile(optimizer=opt,
                  loss=[contrastive_loss, 'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[0.05, 0.5, 0.5])
                  # loss=[contrastive_loss])
    return model


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    from keras.utils.visualize_util import plot

    nb_classes = 1122

    model_file = '/data/liubo/face/annotate_face_model/deepid2_%d.model' %(nb_classes)
    weight_file = '/data/liubo/face/annotate_face_model/deepid2_%d.weight' %(nb_classes)

    model = build_deepid2_model(input_shape=((3, 150, 150)), nb_classes=nb_classes)

    open(model_file, 'w').write(model.to_json())
    model.save_weights(weight_file, overwrite=True)

    model.summary()
    plot(model, to_file="deepid2.png", show_shapes=True)
