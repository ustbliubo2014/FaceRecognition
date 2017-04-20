# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: original_fine_tune_fc7.py
@time: 2016/8/18 12:03
@contact: ustb_liubo@qq.com
@annotation: original_fine_tune_fc7
"""
import sys
import logging
from logging.config import fileConfig
import os
import msgpack_numpy
from keras.models import Model, model_from_json
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from extract_feature import load_data
from conf import *
import keras.backend as K

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def mlp_net(feature_dim, class_num):
    input_layer = Input(shape=(feature_dim,), name='input')
    hidden_layer = Dense(2048, activation='relu', name='hidden_layer1')(input_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)
    hidden_layer = Dense(1024, activation='relu', name='hidden_layer2')(hidden_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)
    classify_layer = Dense(class_num, activation='softmax', name='classify_layer')(hidden_layer)
    # classify_layer = Dense(class_num, activation='softmax', name='classify_layer')(input_layer)
    model = Model(input=[input_layer], output=[classify_layer])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


def load_model(layer_index=-3):
    # model_file = '/data/liubo/face/vgg_face_dataset/model/originalimages_fc7.model'
    # weight_file = '/data/liubo/face/vgg_face_dataset/model/originalimages_fc7.weight'
    model_file = '/data/liubo/face/vgg_face_dataset/model/original_deep_face_180.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/original_deep_face_180.weight'
    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load weights'
        model.load_weights(weight_file)
        get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                 [model.layers[layer_index].get_output_at(False)])
        return model, get_Conv_FeatureMap


if __name__ == '__main__':
    data, label = load_data(originalimages_fc7_data_path)
    # data = np.reshape(data, newshape=(data.shape[0], data.shape[1]))
    class_num = len(set(label))
    model = mlp_net(feature_dim=data.shape[1], class_num=class_num)
    label = np_utils.to_categorical(label, class_num)
    model_file = '/data/liubo/face/vgg_face_dataset/model/originalimages_fc7.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/originalimages_fc7.weight'

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1)
    print X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, monitor='val_acc', save_best_only=True)
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=30, verbose=0, shuffle=True, validation_data=(X_test, Y_test),
              callbacks=[checkpointer])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score
    model.save_weights(weight_file, overwrite=True)
    open(model_file,'w').write(model.to_json())

