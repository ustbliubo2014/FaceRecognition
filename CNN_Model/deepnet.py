# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: deepnet.py
@time: 2016/8/17 16:39
@contact: ustb_liubo@qq.com
@annotation: deepnet
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
import os
from keras.optimizers import Adagrad, Adam, SGD
from keras import backend as K
from keras.layers.convolutional import ZeroPadding2D
import msgpack_numpy
from sklearn.cross_validation import train_test_split
import pdb

# global constants
NB_CLASS = 181  # number of classes
DIM_ORDERING = 'th'  # 'th' (channels, width, height) or 'tf' (width, height, channels)
WEIGHT_DECAY = 0.0005  # L2 regularization factor
USE_BN = True  # whether to use batch normalization


def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    return x


CONCAT_AXIS = 1


def deep_net(pic_shape, nb_classes):
    input_layer = Input(shape=pic_shape)
    # 128x128 -> 64x64
    conv1 = conv2D_bn(input_layer, 64, 3, 3, subsample=(1, 1), border_mode='same')
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv1)
    # 64x64 -> 32x32
    conv2 = conv2D_bn(pool1, 128, 3, 3, subsample=(1, 1), border_mode='same')
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv2)
    # 32x32 -> 16x16
    conv3 = conv2D_bn(pool2, 256, 3, 3, subsample=(1, 1), border_mode='same')
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv3)
    # 16x16 -> 8x8
    conv4 = conv2D_bn(pool3, 512, 3, 3, subsample=(1, 1), border_mode='same')
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv4)
    # 8x8 -> 8x8
    conv5 = conv2D_bn(pool4, 512, 3, 3, subsample=(1, 1), border_mode='same')
    #
    flatten = merge([conv5, pool4], mode='concat', concat_axis=CONCAT_AXIS)
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
    pdb.set_trace()
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if not os.path.exists(model_file):
        model = deep_net(input_shape, nb_classes)
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        open(model_file,'w').write(model.to_json())
    else:
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        # opt = SGD()
        # opt = RMSprop()
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    if os.path.exists(weight_file):
        print 'load_weights'
        model.load_weights(weight_file)


    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
    datagen.fit(X_train, augment=False)
    nb_epoch = 500
    batch_size = 128
    Y_predict_batch = model.predict(X_test, batch_size=batch_size, verbose=1)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(Y_predict_batch, axis=1))
    test_acc = np.min([test_acc, 0.7])
    last_crps = test_acc
    length = X_train.shape[0]
    shuffle_list = range(length)
    print('last_crps :', last_crps)
    this_patience = 0
    patience = 10

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print('Training...')
        progbar = generic_utils.Progbar(length)
        sample_num = 0
        # 每次手动shuffle

        np.random.shuffle(shuffle_list)
        X_train = X_train[shuffle_list]
        y_train = y_train[shuffle_list]
        for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('train loss', loss)])
            sample_num += X_batch.shape[0]
            if sample_num >= X_train.shape[0]:
                break

        print('Testing...')
        Y_predict_batch = model.predict(X_test, batch_size=batch_size, verbose=1)
        test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(Y_predict_batch, axis=1))
        Y_train_preidct_batch = model.predict(X_train, batch_size=batch_size, verbose=1)
        train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(Y_train_preidct_batch, axis=1))
        print ('train_acc :', train_acc,  'test acc', test_acc)
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


def extract_feature(model_file, weight_file):
    print 'model_file :', model_file
    print 'weight_file :', weight_file
    model = model_from_json(open(model_file, 'r').read())
    model.load_weights(weight_file)
    get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                     [model.layers[-2].get_output_at(False)])
    return model, get_Conv_FeatureMap


if __name__ == '__main__':
    model_file = '/data/liubo/face/vgg_face_dataset/model/originalimages.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/originalimages.weight'
    # extract_feature(model_file, weight_file)
    # model = deep_net(pic_shape=(3, 128, 128), nb_classes=NB_CLASS)
    # model.compile('rmsprop', 'categorical_crossentropy')

    model_data, model_label = msgpack_numpy.load(open('/data/liubo/face/originalimages/originalimages_model.p', 'rb'))
    model_data = np.transpose(model_data, (0, 3, 1, 2))
    X_train, X_test, y_train, y_test = train_test_split(model_data, model_label, test_size=0.1)

    train_valid_model(X_train, y_train, X_test, y_test, NB_CLASS, model_file, weight_file)
