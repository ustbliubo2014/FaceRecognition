# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: model_trans.py
@time: 2016/7/6 9:46
@contact: ustb_liubo@qq.com
@annotation: model_trans: 相同输入维度,不同输出维度,将就的模型的conv参数赋值给新的[卷积层相同]
"""


import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
import numpy as np
import pdb
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Input, Lambda, merge, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import os
from time import time
import theano
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
from keras import backend as K


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='model_trans.log',
                    filemode='w')

def load_old_model(model_file, weight_file):
    print 'load old model'
    model = model_from_json(open(model_file, 'r').read())
    model.load_weights(weight_file)
    return model


def load_new_model(model_file):
    print 'load new model'
    model = model_from_json(open(model_file, 'r').read())
    return model


def main_copy_weight():
    # 需要先初始化新的网络,再将原来网络的权值加入到新的网络中
    # 两个模型的网络结构完全相同

    old_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.small.rgb.nose.deepid.model'
    old_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.small.rgb.nose.deepid.weight'
    new_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.model'
    new_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.weight'
    old_model = load_old_model(old_model_file, old_weight_file)
    new_model = load_new_model(new_model_file)
    new_model.layers[1] = old_model.layers[1]
    new_model.save_weights(new_weight_file, overwrite=True)


def main_copy_cnn_weight():
    # 除了softmax层外,复制参数
    old_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
    old_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'
    # new_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.model'
    # new_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.weight'
    old_model = load_old_model(old_model_file, old_weight_file)
    # new_model = load_new_model(new_model_file)
    # new_model.layers[1] = old_model.layers[1]
    # new_model.save_weights(new_weight_file, overwrite=True)
    pdb.set_trace()


def model_trans(trans_index):
    # trans_index 下的权值
    # 将一个网络底层的一些权值赋给另一个新的网络
    old_model_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.model'
    new_weight_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.weight'
    print 'load old model'
    old_model = model_from_json(open(old_model_file, 'r').read())
    opt = Adam()
    old_model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    print 'load old weights'
    old_model.load_weights(new_weight_file)

    print 'new model'
    new_model_file = '/data/liubo/face/vgg_face_dataset/model/deepface_test.model'
    new_weight_file = '/data/liubo/face/vgg_face_dataset/model/deepface_test.weight'
    new_model = model_from_json(open(new_model_file, 'r').read())
    opt = Adam()
    new_model.compile(optimizer=opt, loss=['categorical_crossentropy'])


    for i, layer in enumerate(old_model.layers[0:trans_index]):
        if layer.__class__ in [Convolution2D, Dense]:
            new_model.layers[i].set_weights(old_model.layers[i].get_weights())
    new_model.save_weights(new_weight_file, overwrite=True)

if __name__ == '__main__':
    # main_copy_cnn_weight()
    model_trans(trans_index=14)
    pass
