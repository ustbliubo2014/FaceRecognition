# -*-coding:utf-8 -*-
__author__ = 'liubo-it'


import numpy as np
import pdb
np.random.seed(1337)  # for reproducibility
import random
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, merge, Dense, BatchNormalization
from keras.layers.core import Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
import msgpack
from scipy.misc import imread, imresize
from time import time


def euclidean_distance(vects):
    # 计算距离
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """
        对比损失,参考paper http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices, nb_classes):
    """
        生成对比样本, 同时还要保留原来的data和label,因为softmax需要data和label
    """
    pairs = []
    labels = []
    # 左右样本拆开
    first_datas = []
    first_labels = []
    second_datas = []
    second_labels = []
    # 保证每个类别的正负样本数都一样
    n = min([len(digit_indices[d]) for d in range(nb_classes)]) - 1
    # pdb.set_trace()
    for i in range(n):
        for d in range(nb_classes):
            # pdb.set_trace()
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            # 这个是生成的正样本(z1,z2属于同一个类)
            pairs += [[x[z1], x[z2]]]
            first_datas.append(x[z1])
            first_labels.append(d)
            second_datas.append(x[z2])
            second_labels.append(d)
            inc = random.randrange(1, nb_classes)
            dn = (d + inc) % nb_classes
            # 生成的负样本(z1,z2不是一个类)
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            first_datas.append(x[z1])
            first_labels.append(d)
            second_datas.append(x[z2])
            second_labels.append(dn)
            labels += [1, 0]
            # 第一次的z1和第二次的d1是同一类的样本; 第一次的z2和第二次的z2不是同一类的样本
    return np.array(pairs), np.array(labels), np.asarray(first_datas), np.asarray(first_labels), \
                    np.asarray(second_datas), np.asarray(second_labels)


def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


def create_pair_data(X_train, y_train, X_test, y_test, nb_classes):
    # pdb.set_trace()
    digit_indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
    tr_pairs, tr_y, X_train_first, y_train_first, X_train_second, y_train_second = \
        create_pairs(X_train, digit_indices, nb_classes)
    digit_indices = [np.where(y_test == i)[0] for i in range(nb_classes)]
    te_pairs, te_y, X_test_first, y_test_first, X_test_second, y_test_second = \
        create_pairs(X_test, digit_indices, nb_classes)
    return tr_pairs, tr_y, X_train_first, y_train_first, X_train_second, y_train_second,  \
           te_pairs, te_y, X_test_first, y_test_first, X_test_second, y_test_second


def create_deepId_network(input_shape):
    print 'input_shape :', input_shape
    input_layer = Input(shape=input_shape)
    layer1 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(input_layer)
    layer1 = BatchNormalization()(layer1)
    # layer1 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer1)
    # layer1 = BatchNormalization()(layer1)
    # layer1 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer1)
    # layer1 = BatchNormalization()(layer1)
    layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)
    layer2 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer1)
    layer2 = BatchNormalization()(layer2)
    # layer2 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer2)
    # layer2 = BatchNormalization()(layer2)
    # layer2 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer2)
    # layer2 = BatchNormalization()(layer2)
    layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)
    layer3 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer2)
    layer3 = BatchNormalization()(layer3)
    # layer3 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer3)
    # layer3 = BatchNormalization()(layer3)
    layer3 = MaxPooling2D(pool_size=(2, 2))(layer3)
    layer4 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer3)
    layer4 = BatchNormalization()(layer4)
    # layer4 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same')(layer4)
    # layer4 = BatchNormalization()(layer4)
    layer3_flatten = Flatten()(layer3)
    layer4_flatten = Flatten()(layer4)
    flatten_layer = merge(inputs=[layer3_flatten, layer4_flatten], mode='concat')
    # dense = Dense(2048, activation='relu')(flatten_layer) # 在softmax前增加一个隐含层
    dense = Dense(2048, activation='sigmoid')(flatten_layer) # 在softmax前增加一个隐含层
    model = Model(input=[input_layer], output=[dense])
    # model = Model(input=[input_layer], output=[flatten_layer])
    return model


def build_deepid_model(create_base_network, input_shape, nb_classes):
    print('building deepid model')
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    processed_a = base_network(input_a)
    pred_a = Dense(nb_classes, activation='softmax')(processed_a)
    model = Model(input=[input_a], output=[pred_a])
    opt = SGD(lr=0.01)
    # opt = Adagrad()
    # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    return model