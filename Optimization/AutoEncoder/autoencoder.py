# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: autoencoder.py
@time: 2016/11/29 16:52
@contact: ustb_liubo@qq.com
@annotation: autoencoder
"""
import sys
import logging
from logging.config import fileConfig
import os
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import msgpack_numpy
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json
from keras.optimizers import Adadelta, SGD
import pdb
import traceback
import keras.backend as K

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def build_autoencoder():
    input_img = Input(shape=(feature_dim,))
    encoded = Dense(hidden_dim, activation='sigmoid')(input_img)
    decoded = Dense(feature_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(hidden_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, encoder, decoder


def train():
    data = msgpack_numpy.load(open('all_feature.p', 'rb'))
    weight_file = '/data/liubo/face/annotate_face_model/skyeye_face_autoencoder.weight'
    model_file = '/data/liubo/face/annotate_face_model/skyeye_face_autoencoder.model'
    train_data, valid_data = train_test_split(data, test_size=0.15)
    if os.path.exists(model_file) and os.path.exists(weight_file):
        autoencoder = model_from_json(open(model_file, 'r').read())
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.load_weights(weight_file)
    else:
        autoencoder, encoder, decoder = build_autoencoder()
        open(model_file, 'w').write(autoencoder.to_json())
    autoencoder.fit(train_data, train_data, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(valid_data, valid_data))
    autoencoder.save_weights(weight_file)


def valid():
    data = msgpack_numpy.load(open('all_feature.p', 'rb'))
    weight_file = '/data/liubo/face/annotate_face_model/skyeye_face_autoencoder.weight'
    model_file = '/data/liubo/face/annotate_face_model/skyeye_face_autoencoder.model'
    autoencoder =  model_from_json(open(model_file, 'r').read())
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.load_weights(weight_file)
    get_Conv_FeatureMap = K.function([autoencoder.layers[0].get_input_at(False), K.learning_phase()],
                                     [autoencoder.layers[-2].get_output_at(False)])
    feature_vector = get_Conv_FeatureMap([data[0:3], 0])


if __name__ == '__main__':
    feature_dim = 256  # 人脸特征(隐含层和输入层维度相同)
    hidden_dim = 256
    train()
    valid()
