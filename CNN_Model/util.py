# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: util.py
@time: 2016/8/11 17:00
@contact: ustb_liubo@qq.com
@annotation: util
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
import keras.backend as K
import random
from random import randint
import traceback
import pdb
from sklearn.cross_validation import train_test_split

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')
same_person_id = 0
no_same_person_id = 1


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


def create_pairs(x, digit_indices, class_num):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(class_num)]) - 1
    for d in range(class_num):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, class_num)
            dn = (d + inc) % class_num
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def create_pair_data(person_feature_dic, per_sample_num=50):
    # 每个人生成10张正样本, 10张负样本
    person_list = person_feature_dic.keys()
    person_num = len(person_list)
    pair_list = []
    for person in person_feature_dic:
        try:
            this_person_feature_list = person_feature_dic.get(person)
            this_person_feature_list = map(lambda x:np.reshape(x, newshape=(x.shape[1])), this_person_feature_list)
            path_num = len(this_person_feature_list)
            if path_num < 10:
                continue
            # print person.decode('gbk'),
            count = 0
            # 找10张不一样的
            while count < per_sample_num:
                other_person = person_list[randint(0, person_num-1)]
                if other_person == person:
                    continue
                other_person_feature = person_feature_dic.get(other_person)
                other_person_feature = map(lambda x:np.reshape(x, newshape=(x.shape[1])), other_person_feature)
                if len(other_person_feature) < 1 or len(this_person_feature_list) < 1:
                    continue
                count += 1
                pair_list.append((
                    this_person_feature_list[randint(0, path_num-1)],
                    other_person_feature[randint(0, len(other_person_feature)-1)],
                    no_same_person_id
                    ))
            # 找10张一样的
            count = 0
            has_find = set()
            while count < per_sample_num:
                first_index = randint(0, path_num-1)
                second_index = randint(0, path_num-1)
                if first_index == second_index:
                    continue
                if first_index > second_index:
                    first_index, second_index = second_index, first_index
                    if (first_index, second_index) in has_find:
                        continue
                    else:
                        has_find.add((first_index, second_index))
                        pair_list.append((
                            this_person_feature_list[first_index],
                            this_person_feature_list[second_index],
                            same_person_id
                        ))
                        count += 1
        except:
            traceback.print_exc()
    # print
    np.random.shuffle(pair_list)
    data = map(lambda x: [x[0], x[1]], pair_list)
    label = map(lambda x: x[2], pair_list)
    data = np.asarray(data)
    label = np.asarray(label)
    return data, label


def load_label_data(person_feature_dic):
    # 按label读入所有样本
    all_data= []
    all_label = []
    current_label = 0
    for person in person_feature_dic:
        try:
            this_person_feature_list = person_feature_dic.get(person)
            this_person_feature_list = map(lambda x: np.reshape(x, newshape=(x.shape[1])), this_person_feature_list)
            path_num = len(this_person_feature_list)
            if path_num < 10:
                continue
            this_current_label_list = [current_label] * path_num
            all_data.extend(this_person_feature_list)
            all_label.extend(this_current_label_list)
            current_label += 1
        except:
            traceback.print_exc()
    all_data = np.asarray(all_data)
    all_data = all_data / np.max(all_data)
    all_label = np.asarray(all_label)
    train_data, valid_data, train_label, valid_label = train_test_split(all_data, all_label, test_size=0.5)
    return train_data, valid_data, train_label, valid_label
