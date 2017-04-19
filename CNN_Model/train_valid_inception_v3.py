# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: train_valid_inception_v3.py
@time: 2016/9/6 16:39
@contact: ustb_liubo@qq.com
@annotation: train_valid_inception_v3
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

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


one_load_sample_num = 6400
batch_size = 32
nb_epoch = 500

datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images


def train_model(train_path_list, model, nb_classes, pic_shape):
    length = len(train_path_list)
    train_load_num = length / one_load_sample_num
    np.random.shuffle(train_path_list)
    progbar = generic_utils.Progbar(length)
    for train_load_index in range(train_load_num):
        try:
            X_train, y_train = load_data_from_list(train_path_list[
                            train_load_index*one_load_sample_num: (train_load_index+1)*one_load_sample_num], pic_shape)
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            # print X_train.shape, Y_train.shape
            # model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, validation_split=0.1, nb_epoch=2)
            datagen.fit(X_train, augment=False)
            sample_num = 0
            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size):
                loss = model.train_on_batch(X_batch, [Y_batch, Y_batch])
                # progbar中显示的是均值
                progbar.add(X_batch.shape[0], values=[('train loss', loss[0])])
                sample_num += X_batch.shape[0]
                if sample_num >= X_train.shape[0]:
                    break
        except:
            traceback.print_exc()
            continue


def valid_model(valid_path_list, model, nb_classes, pic_shape):
    np.random.shuffle(valid_path_list)
    # 不用测试所有的样本, 只需随机测试一部分就可以
    valid_load_num = min(2, len(valid_path_list) / one_load_sample_num)
    # valid_load_num = len(valid_path_list) / one_load_sample_num
    all_acc = []
    for valid_load_index in range(valid_load_num):
        X_valid, y_valid = load_data_from_list(valid_path_list[
                        valid_load_index*one_load_sample_num:(valid_load_index+1)*one_load_sample_num], pic_shape)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes=nb_classes)
        Y_predict_batch = model.predict(X_valid, batch_size=batch_size, verbose=1)
        test_acc = accuracy_score(np.argmax(Y_valid, axis=1),
                                  np.argmax((Y_predict_batch[0] + Y_predict_batch[1]) / 2.0, axis=1))
        all_acc.append(test_acc)


    mean_acc = np.min(all_acc)
    return mean_acc


def train_valid_model(train_path_list, valid_path_list, pic_shape, nb_classes, model_file, weight_file):
    print 'load model'
    model = model_from_json(open(model_file, 'r').read())
    opt = RMSprop()
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
        print 'this_acc :', this_acc, 'last_acc :', last_acc
        if this_acc > last_acc:
            model.save_weights(weight_file, overwrite=True)
            print ('save_model')
            last_acc = this_acc


if __name__ == '__main__':
    # 提前确定好NB_CLASS（36718）, 先将model和weight写入文件,直接从文件中加载model
    model_file = '/data/liubo/face/MS-Celeb_face_model/inception_v3_36718.model'
    weight_file = '/data/liubo/face/MS-Celeb_face_model/inception_v3_36718.weight'
    nb_classes = 36718
    pic_shape = (299, 299, 3) # inception_v3的shape
    train_valid_sample_list_file = '/data/liubo/face/MS-Celeb_face_list/sample_list_all_shuffle.p'
    (train_sample_list, valid_sample_list) = msgpack_numpy.load(open(train_valid_sample_list_file, 'rb'))
    train_valid_model(train_sample_list, valid_sample_list, pic_shape, nb_classes, model_file, weight_file)
