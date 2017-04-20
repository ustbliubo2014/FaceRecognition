# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: util.py
@time: 2016/8/18 10:20
@contact: ustb_liubo@qq.com
@annotation: util
"""
import sys
import logging
from logging.config import fileConfig
import os
from scipy.misc import imread, imresize
from keras.models import model_from_json
from keras.optimizers import Adam
import keras.backend as K
import pdb
from keras import regularizers
from keras.layers import BatchNormalization, Convolution2D

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


# global constants
CONCAT_AXIS = 1
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


def read_one_rgb_pic(pic_path, pic_shape, avg):
    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def load_model(output_layer_index):
    model_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.weight'
    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load weights'
        model.load_weights(weight_file)
        get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                 [model.layers[output_layer_index].get_output_at(False)])
        return model, get_Conv_FeatureMap
    else:
        return None, None


if __name__ == '__main__':
    model, get_Conv_FeatureMap = load_model(-1)
    model.summary()
