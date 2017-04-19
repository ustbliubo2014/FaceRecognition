# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: train_valid_deepid2.py
@time: 2016/9/30 15:51
@contact: ustb_liubo@qq.com
@annotation: train_valid_deepid2
"""
import sys
import logging
from logging.config import fileConfig
import os
from time import time
from keras.utils import np_utils, generic_utils
from DeepId import create_pairs, contrastive_loss
from DeepId2 import build_deepid2_model
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from load_data import load_data_from_list
import traceback
import pdb
import msgpack_numpy
from keras.models import  model_from_json
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from sklearn.metrics import accuracy_score

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

one_load_sample_num = 6400
batch_size = 6400
nb_epoch = 500
small_batch_size = 32

datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images


def valid_model(valid_path_list, model, nb_classes, pic_shape):
    np.random.shuffle(valid_path_list)
    # 不用测试所有的样本, 只需随机测试一部分就可以
    valid_load_num = min(2, len(valid_path_list) / one_load_sample_num)

    all_loss = []

    for valid_load_index in range(valid_load_num):
        X_valid, y_valid = load_data_from_list(valid_path_list[
                        valid_load_index*one_load_sample_num:(valid_load_index+1)*one_load_sample_num], pic_shape)

        for X_batch, y_valid in datagen.flow(X_valid, y_valid, batch_size=batch_size):
            valid_pairs, valid_label, X_valid_first, y_valid_first, X_valid_second, \
            y_valid_second = create_pairs(X_valid, y_valid)
            pair_num = valid_pairs.shape[0]

            this_batch_num = pair_num / small_batch_size
            for k in range(this_batch_num):
                loss = model.predict_on_batch(
                    [
                        valid_pairs[k * small_batch_size:(k + 1) * small_batch_size, 0],
                        valid_pairs[k * small_batch_size:(k + 1) * small_batch_size, 1]
                    ]
                )
                all_loss.append(loss[2])

    mean_loss = np.min(all_loss)
    return mean_loss


def train_model(train_path_list, model, nb_classes, pic_shape):
    length = len(train_path_list)
    train_load_num = length / one_load_sample_num
    np.random.shuffle(train_path_list)
    progbar = generic_utils.Progbar(length)
    for train_load_index in range(train_load_num):
        try:
            X_train, y_train = load_data_from_list(train_path_list[
                            train_load_index*one_load_sample_num: (train_load_index+1)*one_load_sample_num], pic_shape)

            datagen.fit(X_train, augment=False)
            for X_batch, y_train in datagen.flow(X_train, y_train, batch_size=batch_size):
                train_pairs, train_label, X_train_first, y_train_first, X_train_second, \
                        y_train_second = create_pairs(X_train, y_train)
                y_train_first = np_utils.to_categorical(y_train_first, nb_classes)
                y_train_second = np_utils.to_categorical(y_train_second, nb_classes)
                pair_num = train_pairs.shape[0]

                this_batch_num = pair_num / small_batch_size
                for k in range(this_batch_num):
                    loss = model.train_on_batch(
                        [
                            train_pairs[k*small_batch_size:(k+1)*small_batch_size, 0],
                            train_pairs[k*small_batch_size:(k+1)*small_batch_size, 1]
                        ],
                        [
                            train_label[k*small_batch_size:(k+1)*small_batch_size],
                            y_train_first[k*small_batch_size:(k+1)*small_batch_size, :],
                            y_train_second[k*small_batch_size:(k+1)*small_batch_size, :]
                        ]
                    )
                    print loss
                    # progbar.add(X_batch.shape[0], values=[('train loss', loss[2])])
                break
        except:
            traceback.print_exc()
            continue



def train_valid_model(train_path_list, valid_path_list, pic_shape, nb_classes, model_file, weight_file):
    print 'load model'
    model = model_from_json(open(model_file, 'r').read())
    # opt = RMSprop()
    # opt = Adam()
    opt = SGD(momentum=0.9, lr=0.0001)
    model.compile(optimizer=opt,
                  loss=[contrastive_loss, 'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[0.02, 0.49, 0.49])
    print 'load weights'
    model.load_weights(weight_file)

    # last_loss = valid_model(valid_path_list, model, nb_classes, pic_shape)
    # print 'first_loss :', last_loss
    for epoch_index in range(nb_epoch):
        print('-'*40)
        print('Training ', 'current epoch :', epoch_index, 'all epcoh :', nb_epoch)
        train_model(train_path_list, model, nb_classes, pic_shape)
        model.save_weights(weight_file, overwrite=True)
        # this_loss = valid_model(valid_path_list, model, nb_classes, pic_shape)
        # print 'this_loss :', this_loss, 'last_loss :', last_loss
        # if this_loss < last_loss:
        #     model.save_weights(weight_file, overwrite=True)
        #     print ('save_model')
        #     last_loss = this_loss


if __name__ == '__main__':
    # 提前确定好NB_CLASS（1122）, 先将model和weight写入文件,直接从文件中加载model
    nb_classes = 1122

    model_file = '/data/liubo/face/annotate_face_model/deepid2_%d.model' %(nb_classes)
    weight_file = '/data/liubo/face/annotate_face_model/deepid2_%d.weight' %(nb_classes)

    pic_shape = (150, 150, 3) # inception_v3的shape
    train_valid_sample_list_file = '/data/liubo/face/all_pic_data/train_valid_sample_list.p'
    (train_sample_list, valid_sample_list) = msgpack_numpy.load(open(train_valid_sample_list_file, 'rb'))

    train_valid_model(train_sample_list, valid_sample_list, pic_shape, nb_classes, model_file, weight_file)


