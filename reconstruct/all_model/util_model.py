# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId.py
@time: 2016/7/27 15:39
@contact: ustb_liubo@qq.com
@annotation: util_model
"""

from keras.models import (
    Sequential,
    Graph
)
from keras.layers.core import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Merge
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.normalization import BatchNormalization
from model_conf import *
from keras import regularizers


def conv2D_bn(last_layer, nb_filter, nb_row=nb_conv, nb_col=nb_conv,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=dim_ordering):
    '''
        Utility function to apply to a tensor a module conv + BN
        with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None
    last_layer = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(last_layer)
    if batch_norm:
        last_layer = BatchNormalization()(last_layer)
    return last_layer
