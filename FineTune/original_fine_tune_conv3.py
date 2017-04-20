# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: model.py
@time: 2016/8/18 11:14
@contact: ustb_liubo@qq.com
@annotation: model
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from util import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
import os
from keras.optimizers import Adagrad, Adam, SGD
from keras import backend as K
import msgpack_numpy
from sklearn.cross_validation import train_test_split
import pdb
from conf import *
from extract_feature import load_data
from collections import Counter


def originalimages_fine_tune(pic_shape, nb_classes):
    '''
    :param pic_shape:输入向量的shape,由于是微调,所以shape一般是(256, 16, 16) [(filter_num, new_pic_shape, new_pic_shape)]
    :param nb_classes:
    :return:
    '''
    input_layer = Input(shape=pic_shape)
    # 16x16 -> 8x8
    conv1 = conv2D_bn(input_layer, 512, 3, 3, subsample=(1, 1), border_mode='same')
    conv1 = conv2D_bn(conv1, 512, 3, 3, subsample=(1, 1), border_mode='same')
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv1)
    # 8x8 -> 8x8
    conv2 = conv2D_bn(pool1, 512, 3, 3, subsample=(1, 1), border_mode='same')
    conv2 = conv2D_bn(conv2, 512, 3, 3, subsample=(1, 1), border_mode='same')

    flatten = merge([pool1, conv2], mode='concat', concat_axis=CONCAT_AXIS)
    flatten = Flatten()(flatten)
    flatten = Dropout(0.5)(flatten)

    fc1 = Dense(2048, activation='relu')(flatten)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    preds = Dense(nb_classes, activation='softmax')(fc2)

    model = Model(input=input_layer, output=preds)
    print model.summary()
    return model


def train_valid_model(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file):
    input_shape = X_train.shape[1:]
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if not os.path.exists(model_file):
        model = originalimages_fine_tune(input_shape, nb_classes)
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        open(model_file,'w').write(model.to_json())
    else:
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    if os.path.exists(weight_file):
        print 'load_weights'
        model.load_weights(weight_file)


    nb_epoch = 500
    batch_size = 128
    Y_predict_batch = model.predict(X_test, batch_size=batch_size, verbose=1)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(Y_predict_batch, axis=1))
    test_acc = np.min([test_acc, 0.7])
    last_crps = test_acc
    print('last_crps :', last_crps)
    this_patience = 0
    patience = 10

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print('Training...')

        model.fit(X_train, y_train, batch_size=batch_size, shuffle=True, nb_epoch=1)
        print('Testing...')
        Y_predict_batch = model.predict(X_test, batch_size=batch_size, verbose=1)
        print Counter(np.argmax(Y_predict_batch, axis=1))
        test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(Y_predict_batch, axis=1))
        Y_train_preidct_batch = model.predict(X_train, batch_size=batch_size, verbose=1)
        train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(Y_train_preidct_batch, axis=1))
        print ('train_acc :', train_acc, 'test acc', test_acc)
        if last_crps < test_acc:
            this_patience = 0
            model.save_weights(weight_file, overwrite=True)
            print ('save_model')
            last_crps = test_acc
        else:
            if this_patience >= patience:
                break
            else:
                this_patience = 1


if __name__ == '__main__':
    feature, label = load_data(originalimages_conv3_data_path)
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.1)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    model_file = '/data/liubo/face/vgg_face_dataset/model/originalimages_fine_tune.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/originalimages_fine_tune.weight'
    train_valid_model(X_train, y_train, X_test, y_test, nb_classes=181, model_file=model_file, weight_file=weight_file)
