# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: deepface.py
@time: 2016/8/26 16:43
@contact: ustb_liubo@qq.com
@annotation: deepface
"""
import sys
import logging
from logging.config import fileConfig
import os
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
import msgpack_numpy

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


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


def deepface(pic_shape, class_num):
    input_layer = Input(shape=pic_shape)
    # 使用DeepFace的模型的底层网络要和DeepFace的设置一致(图片大小可以改变) -- 取pool2后的输出进行微调
    # 将DeepFace之后的输出作为
    # # 128x128 -> 64x64
    # conv1 = conv2D_bn(input_layer, 64, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # conv1 = conv2D_bn(conv1, 64, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # pool1 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv1)
    # # 64x64 -> 32x32
    # conv2 = conv2D_bn(pool1, 128, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # conv2 = conv2D_bn(conv2, 128, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # pool2 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv2)
    # # 32x32 -> 16x16
    # conv3 = conv2D_bn(pool2, 256, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # conv3 = conv2D_bn(conv3, 256, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # conv3 = conv2D_bn(conv3, 256, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # pool3 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv3)
    # # 16x16 -> 8x8
    # conv4 = conv2D_bn(pool3, 512, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # conv4 = conv2D_bn(conv4, 512, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # conv4 = conv2D_bn(conv4, 512, 3, 3, subsample=(1, 1), border_mode='same', batch_norm=False, weight_decay=0)
    # pool4 = MaxPooling2D((2, 2), strides=(2, 2), dim_ordering=DIM_ORDERING)(conv4)

    # flatten = Flatten()(pool4)
    flatten = Flatten()(input_layer)
    flatten = Dropout(0.5)(flatten)

    fc5 = Dense(4096, activation='relu')(flatten)
    fc5 = Dropout(0.5)(fc5)
    fc6 = Dense(4096, activation='relu')(fc5)
    fc6 = Dropout(0.5)(fc6)

    preds = Dense(class_num, activation='softmax')(fc6)

    model = Model(input=input_layer, output=preds)
    print model.summary()
    return model


def train_valid(train_data, valid_data, train_label, valid_label, model_file, weight_file):
    pic_shape = train_data.shape[1:]
    class_num = max(max(valid_label), max(train_label)) + 1

    train_label = np_utils.to_categorical(train_label, class_num)
    valid_label = np_utils.to_categorical(valid_label, class_num)

    if not os.path.exists(model_file):
        model = deepface(pic_shape, class_num)
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
    datagen.fit(train_data, augment=False)
    nb_epoch = 500
    batch_size = 128
    Y_predict_batch = model.predict(valid_data, batch_size=batch_size, verbose=1)
    test_acc = accuracy_score(np.argmax(valid_label, axis=1), np.argmax(Y_predict_batch, axis=1))
    last_crps = test_acc
    length = train_data.shape[0]
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
        train_data = train_data[shuffle_list]
        train_label = train_label[shuffle_list]
        for X_batch, Y_batch in datagen.flow(train_data, train_label, batch_size=batch_size):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('train loss', loss)])
            sample_num += X_batch.shape[0]
            if sample_num >= train_data.shape[0]:
                break

        print('Testing...')
        Y_predict_batch = model.predict(valid_data, batch_size=batch_size, verbose=1)
        test_acc = accuracy_score(np.argmax(valid_label, axis=1), np.argmax(Y_predict_batch, axis=1))
        Y_train_preidct_batch = model.predict(train_data, batch_size=batch_size, verbose=1)
        train_acc = accuracy_score(np.argmax(train_label, axis=1), np.argmax(Y_train_preidct_batch, axis=1))
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


if __name__ == '__main__':
    model = deepface(pic_shape=(512, 7, 7), class_num=168)
    model_file = '/data/liubo/face/vgg_face_dataset/model/deepface_test.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/deepface_test.weight'
    data, label = msgpack_numpy.load(open('/home/liubo-it/FaceRecognization/FineTune/v2/hanlin.p', 'rb'))
    data = np.asarray(data)
    label = np.asarray(label)
    train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.2)
    print train_data.shape, valid_data.shape
    train_valid(train_data, valid_data, train_label, valid_label, model_file, weight_file)
