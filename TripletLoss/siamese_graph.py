# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: siamese_graph.py
@time: 2016/9/9 14:12
@contact: ustb_liubo@qq.com
@annotation: siamese_graph
"""
import sys
import logging
from logging.config import fileConfig
import os
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K
import numpy as np
from sklearn.cross_validation import train_test_split
import msgpack_numpy
from keras.datasets import mnist
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


nb_class = 737
input_dim = 4096


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# 数据大时,每次读入部分数据(sample_list)
def create_pairs(x, digit_indices):
    '''
        Positive and negative pair creation. Alternates between positive and negative pairs.
        :param digit_indices : 每个类的样本的index[[1,2,3],[4,5,6],](0类样本在1,2,3;1类样本在4,5,6)
        :param x : data (51999, 4096) [51999个样本,4096维]
        face数据中, 每个人的照片比较少,所以要选择尽可能多的正样本和相同数量的负样本
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(nb_class)]) - 1
    # n : 样本数最少的类的样本数 (每个类选n个正样本, n个负样本)
    for d in range(nb_class):
        this_sample_num = min(20, len(digit_indices[d]))
        # this_sample_num = n
        for i in range(this_sample_num):
            for j in range(i+1, this_sample_num):
                z1, z2 = digit_indices[d][i], digit_indices[d][j]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, nb_class)
                dn = (d + inc) % nb_class
                other_j = random.randrange(0, len(digit_indices[dn]))
                y1, y2 = digit_indices[d][i], digit_indices[dn][other_j]
                pairs += [[x[y1], x[y2]]]
                labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    seq = Sequential()
    seq.add(Dense(1024, input_shape=(input_dim,), activation='tanh'))
    seq.add(Dropout(0.5))
    seq.add(Dense(256, activation='tanh'))
    seq.add(Dropout(0.5))
    seq.add(Dense(128, activation='tanh'))
    return seq


def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


def load_data():
    # the data, shuffled and split between train and test sets
    (data, label) = msgpack_numpy.load(open('/data/pictures_annotate_feature/annotate_data.p', 'rb'))
    digit_indices = [np.where(label == i)[0] for i in range(nb_class)]
    pairs_x, pairs_y = create_pairs(data, digit_indices)
    pairs_x = pairs_x[:10000]
    pairs_y = pairs_y[:10000]
    tr_pairs, te_pairs, tr_y, te_y = train_test_split(pairs_x, pairs_y, test_size=0.1)
    print tr_pairs.shape, te_pairs.shape, tr_y.shape, te_y.shape
    return tr_pairs, te_pairs, tr_y, te_y


def build_siamese_graph():
    base_network = create_base_network(input_dim)
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    tr_pairs, te_pairs, tr_y, te_y = load_data()
    model = build_siamese_graph()

    model_file = '/data/liubo/face/vgg_face_model/annotate_siamese_graph.model'
    open(model_file, 'w').write(model.to_json())
    weight_file = '/data/liubo/face/vgg_face_model/annotate_siamese_graph.weight'
    # if os.path.exists(weight_file):
    #     model.load_weights(weight_file)
    check_pointer = ModelCheckpoint(filepath=weight_file, save_best_only=True)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]],tr_y,
                validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                batch_size=32, nb_epoch=50, callbacks=[check_pointer])
    predict = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(predict, tr_y)
    predict = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(predict, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

