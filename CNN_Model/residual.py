# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: residual.py
@time: 2016/7/18 9:54
@contact: ustb_liubo@qq.com
@annotation: residual
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='residual.log',
                    filemode='a+')
import pdb
from keras import backend as K
import os
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad
)
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# http://arxiv.org/pdf/1512.03385v1.pdf
# 50 Layer resnet
def resnet(pic_shape, nb_classes):
    # input = Input(shape=(3, 224, 224))
    input = Input(shape=pic_shape)

    conv1 = _conv_bn_relu(nb_filter=64, nb_row=3, nb_col=3, subsample=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same")(conv1)

    # Build residual blocks..
    block_fn = _bottleneck
    block1 = _residual_block(block_fn, nb_filters=64, repetations=2, is_first_layer=True)(pool1)
    block2 = _residual_block(block_fn, nb_filters=64, repetations=2)(block1)
    block3 = _residual_block(block_fn, nb_filters=128, repetations=2)(block2)
    # block4 = _residual_block(block_fn, nb_filters=256, repetations=2)(block3)

    # Classifier block
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), border_mode="same")(block3)
    flatten1 = Flatten()(pool2)
    hidden = Dense(output_dim=1024, init="he_normal", activation="sigmoid")(flatten1)
    output = Dense(output_dim=nb_classes, activation='softmax')(hidden)
    model = Model(input=input, output=output)
    return model


def train_valid_model(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file):
    input_shape = X_train.shape[1:]
    # pdb.set_trace()
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if not os.path.exists(model_file):
        model = resnet(input_shape, nb_classes)
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
    batch_size = 512
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
    model_file = '/data/liubo/face/vgg_face_dataset/model/lfw.rgb.residual.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/lfw.rgb.residual.weight'
    extract_feature(model_file, weight_file)

