# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: inception.py
@time: 2016/7/21 19:27
@contact: ustb_liubo@qq.com
@annotation: inception
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
from keras.optimizers import Adagrad, Adam
from keras import backend as K
from keras.layers.convolutional import ZeroPadding2D

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='inception.log',
                    filemode='a+')


# global constants
NB_CLASS = 1000  # number of classes
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


def google_net(pic_shape, nb_classes):
    input_layer = Input(shape=pic_shape)
    x = conv2D_bn(input_layer, 32, 3, 3, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D((2, 2), dim_ordering=DIM_ORDERING)(x)

    branch1x1 = conv2D_bn(x, 64, 1, 1)
    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((2, 2), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 32, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    branch1x1 = conv2D_bn(x, 64, 1, 1)

    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((2, 2), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    aux_logits = AveragePooling2D((2, 2), strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
    aux_logits = conv2D_bn(aux_logits, 512, 3, 3, border_mode='same')
    aux_logits = Flatten()(aux_logits)
    aux_preds = Dense(nb_classes, activation='softmax')(aux_logits)

    branch3x3 = conv2D_bn(x, 192, 1, 1)
    branch3x3 = conv2D_bn(branch3x3, 320, 3, 3, subsample=(2, 2), border_mode='same')

    branch7x7x3 = conv2D_bn(x, 192, 1, 1)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 3, 3, subsample=(2, 2), border_mode='same')

    x_pad = ZeroPadding2D(padding=(1, 1), dim_ordering='th')(x)
    branch_pool = AveragePooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x_pad)
    x = merge([branch3x3, branch7x7x3, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    x = AveragePooling2D((2, 2), strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    preds = Dense(nb_classes, activation='softmax')(x)

    # Define model

    model = Model(input=[input_layer], output=[preds, aux_preds])
    # model.compile('rmsprop', 'categorical_crossentropy')

    return model


def train_valid_model(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file):
    input_shape = X_train.shape[1:]
    # pdb.set_trace()
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if not os.path.exists(model_file):
        model = google_net(input_shape, nb_classes)
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
    # print model.summary()
    return model, get_Conv_FeatureMap


if __name__ == '__main__':
    # model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.300.new_shape.rgb.google_net.model'
    # weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.300.new_shape.rgb.google_net.weight'
    # extract_feature(model_file, weight_file)
    model = google_net(pic_shape=(3, 64, 64), nb_classes=500)
    model.compile('rmsprop', 'categorical_crossentropy')
    model.summary()

