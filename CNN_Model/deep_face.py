# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: deep_face.py
@time: 2016/8/8 18:22
@contact: ustb_liubo@qq.com
@annotation: deep_face
"""
import sys
import logging
from logging.config import fileConfig
import os
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from load_data import load_data_from_list
import pdb
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

one_load_sample_num = 12800
batch_size = 64
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
            print X_train.shape, Y_train.shape
            # model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, validation_split=0.1, nb_epoch=2)
            datagen.fit(X_train, augment=False)
            sample_num = 0
            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size):
                loss = model.train_on_batch(X_batch, Y_batch)
                progbar.add(X_batch.shape[0], values=[('train loss', loss)])
                sample_num += X_batch.shape[0]
                if sample_num >= X_train.shape[0]:
                    break
        except:
            traceback.print_exc()
            continue


def valid_model(valid_path_list, model, nb_classes, pic_shape):
    np.random.shuffle(valid_path_list)
    valid_load_num = len(valid_path_list) / one_load_sample_num
    all_acc = []
    for valid_load_index in range(valid_load_num):
        X_valid, y_valid = load_data_from_list(valid_path_list[
                        valid_load_index*one_load_sample_num:(valid_load_index+1)*one_load_sample_num], pic_shape)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes=nb_classes)
        Y_predict_batch = model.predict(X_valid, batch_size=batch_size, verbose=1)
        test_acc = accuracy_score(np.argmax(Y_valid, axis=1), np.argmax(Y_predict_batch, axis=1))
        all_acc.append(test_acc)

    X_valid, y_valid = load_data_from_list(valid_path_list[valid_load_num*one_load_sample_num:4096], pic_shape)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes=nb_classes)

    Y_predict_batch = model.predict(X_valid, batch_size=batch_size, verbose=1)

    test_acc = accuracy_score(np.argmax(Y_valid, axis=1), np.argmax(Y_predict_batch, axis=1))
    all_acc.append(test_acc)

    mean_acc = np.min(all_acc)
    return mean_acc


def train_valid_model(train_path_list, valid_path_list, pic_shape, nb_classes, model_file, weight_file):
    print 'load model'
    model = model_from_json(open(model_file, 'r').read())
    opt = Adam()
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
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
    pass
