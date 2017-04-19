# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: facenet.py
@time: 2016/8/11 10:14
@contact: ustb_liubo@qq.com
@annotation: facenet : 使用triple loss 训练模型
"""
import sys
import logging
from logging.config import fileConfig
import os
from util import (euclidean_distance, eucl_dist_output_shape,
                  contrastive_loss, create_pairs, load_label_data,
                  compute_accuracy)
import traceback
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import np_utils
from load_data import load_data_from_list
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Dropout, Activation
import msgpack_numpy
from sklearn.metrics import accuracy_score
import pdb
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

one_load_person_num = 10
batch_size = 64
nb_epoch = 500


def build_base_model(feature_dim):
    input_layer = Input(shape=(feature_dim,), name='input')
    hidden_layer = Dense(2048, activation='sigmoid', name='hidden_layer1')(input_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)
    hidden_layer = Dense(1024, activation='sigmoid', name='hidden_layer2')(hidden_layer)
    base_model = Model(input=[input_layer], output=[hidden_layer])
    return base_model


def build_model(feature_dim):
    base_model = build_base_model(feature_dim)
    input_a = Input(shape=(feature_dim,))
    input_b = Input(shape=(feature_dim,))
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=[distance])
    opt = Adam()
    # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
    model.compile(optimizer=opt, loss=[contrastive_loss])
    return model


# 加载数据的时候以人为单位,一次加载100个人的所有图片,在这些图片中生成正样本和负样本
def train_model(person_path_dic, model):
    train_data, valid_data, train_label, valid_label = load_label_data(person_path_dic)
    class_num = len(set(train_label))
    digit_indices = [np.where(train_label == i)[0] for i in range(class_num)]
    train_pair_data, train_pair_label = create_pairs(train_data, digit_indices, class_num)
    digit_indices = [np.where(valid_label == i)[0] for i in range(class_num)]
    valid_pair_data, valid_pair_label = create_pairs(valid_data, digit_indices, class_num)

    model.fit([train_pair_data[:, 0], train_pair_data[:, 1]], train_pair_label, batch_size=128, nb_epoch=5)

    train_pred = model.predict([train_pair_data[:, 0], train_pair_data[:, 1]])
    valid_pred = model.predict([valid_pair_data[:, 0], valid_pair_data[:, 1]])
    clf = LinearSVC()
    clf.fit(train_pred, train_pair_label)
    acc = accuracy_score(clf.predict(valid_pred), valid_pair_label)
    print acc
    # pdb.set_trace()


def train_valid_model(train_person_feature_list_dic, valid_person_feature_list_dic, model_file, weight_file):
    print 'load model'
    model = model_from_json(open(model_file, 'r').read())
    opt = RMSprop()
    model.compile(optimizer=opt, loss=[contrastive_loss])
    print 'load weights'
    model.load_weights(weight_file)

    # train_model(person_feature_list_dic, model)
    # last_acc = valid_model(valid_person_feature_list_dic, model)
    # print 'first_acc :', last_acc
    for epoch_index in range(nb_epoch):
        print('-'*40)
        print('Training ', 'current epoch :', epoch_index, 'all epcoh :', nb_epoch)
        train_model(train_person_feature_list_dic, model)
        # this_acc = valid_model(valid_path_list, model, pic_shape)
        # print 'this_acc :', this_acc, 'last_acc :', last_acc
        # if this_acc > last_acc:
        #     model.save_weights(weight_file, overwrite=True)
        #     print ('save_model')
        #     last_acc = this_acc


if __name__ == '__main__':
    model_file = '/data/liubo/face/vgg_face_dataset/model/facenet.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/facenet.weight'
    model = build_model(feature_dim=4096)
    print model.summary()
    print model.layers[2].summary()
    model.save_weights(weight_file, overwrite=True)
    open(model_file,'w').write(model.to_json())

    person_feature_list_dic = msgpack_numpy.load(open('/data/pictures_annotate_feature/person_feature_list_dic.p', 'rb'))
    train_valid_model(person_feature_list_dic, person_feature_list_dic, model_file, weight_file)
