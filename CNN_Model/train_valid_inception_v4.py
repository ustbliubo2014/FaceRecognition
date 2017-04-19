# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: train_valid_inception_v3.py
@time: 2016/9/6 16:39
@contact: ustb_liubo@qq.com
@annotation: train_valid_inception_v4
"""
import sys
import logging
from logging.config import fileConfig
import os
from load_data import load_data_from_list
import msgpack_numpy
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
import traceback
import pdb
from time import sleep
import ImageAugmenter
from optparse import OptionParser


reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


one_load_sample_num = 12800
batch_size = 32
nb_epoch = 500


def train_model(train_path_list, model, nb_classes, pic_shape):
    length = len(train_path_list)
    train_load_num = length / one_load_sample_num
    np.random.shuffle(train_path_list)
    for train_load_index in range(train_load_num):
        try:
            X_train, y_train = load_data_from_list(train_path_list[
                            train_load_index*one_load_sample_num:
                            (train_load_index+1)*one_load_sample_num], pic_shape, need_augment=True)
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            # 数据的augment放在load_data中实现
            model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, nb_epoch=3)
        except:
            traceback.print_exc()
            continue


def valid_model(valid_path_list, model, nb_classes, pic_shape):
    np.random.shuffle(valid_path_list)
    # 不用测试所有的样本, 只需随机测试一部分就可以
    valid_load_num = min(1, len(valid_path_list) / one_load_sample_num)
    # valid_load_num = len(valid_path_list) / one_load_sample_num
    all_acc = []
    if valid_load_num == 0:
        X_valid, y_valid = load_data_from_list(valid_path_list, pic_shape, need_augment=True)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes=nb_classes)
        Y_predict_batch = model.predict(X_valid, batch_size=batch_size, verbose=1)
        test_acc = accuracy_score(np.argmax(Y_valid, axis=1), np.argmax(Y_predict_batch, axis=1))
        all_acc.append(test_acc)
    else:
        for valid_load_index in range(valid_load_num):
            X_valid, y_valid = load_data_from_list(valid_path_list[
                        valid_load_index*one_load_sample_num:(valid_load_index+1)*one_load_sample_num],
                                                   pic_shape, need_augment=True)
            Y_valid = np_utils.to_categorical(y_valid, nb_classes=nb_classes)
            Y_predict_batch = model.predict(X_valid, batch_size=batch_size, verbose=1)
            test_acc = accuracy_score(np.argmax(Y_valid, axis=1), np.argmax(Y_predict_batch, axis=1))
            all_acc.append(test_acc)
    mean_acc = np.min(all_acc)
    return mean_acc


def train_valid_model(train_path_list, valid_path_list, pic_shape, nb_classes, model_file, weight_file):
    print 'load model'
    model = model_from_json(open(model_file, 'r').read())
    # opt = RMSprop()
    # opt = Adam()
    opt = SGD(momentum=0.9)
    model.compile(opt, 'categorical_crossentropy')
    print 'load weights'
    model.load_weights(weight_file)

    last_acc = valid_model(valid_path_list, model, nb_classes, pic_shape)
    print 'first_acc :', last_acc
    for epoch_index in range(nb_epoch):
        print('-'*40)
        print('Training ', 'current epoch :', epoch_index, 'all epcoh :', nb_epoch)
        train_model(train_path_list, model, nb_classes, pic_shape)
        this_acc = valid_model(valid_path_list, model, nb_classes, pic_shape)
        train_acc = valid_model(train_path_list, model, nb_classes, pic_shape)
        print 'this_acc :', this_acc, 'last_acc :', last_acc, 'train_acc :', train_acc
        if this_acc > last_acc:
            model.save_weights(weight_file, overwrite=True)
            print ('save_model')
            last_acc = this_acc


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-n", "--num_class", dest="num_class", help="classify label num")
    parser.add_option("-m", "--model_file", dest="model_file", help="model file")
    parser.add_option("-w", "--weight_file", dest="weight_file", help="weight file")
    parser.add_option("-l", "--train_valid_sample_list_file", dest="train_valid_sample_list_file",
                      help="train_valid_sample_list_file")

    (options, args) = parser.parse_args()

    model_file = options.model_file
    weight_file = options.weight_file
    nb_classes = int(options.num_class)
    train_valid_sample_list_file = options.train_valid_sample_list_file
    pic_shape = (96, 96, 3)  # inception_v4的shape
    (train_sample_list, valid_sample_list) = msgpack_numpy.load(open(train_valid_sample_list_file, 'rb'))
    print 'len(train_sample_list) :', len(train_sample_list), 'len(valid_sample_list) :', len(valid_sample_list)
    train_valid_model(train_sample_list, valid_sample_list, pic_shape, nb_classes, model_file, weight_file)


