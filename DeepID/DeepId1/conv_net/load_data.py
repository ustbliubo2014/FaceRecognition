#!/usr/bin/env python
# -*- coding:utf8 -*-

import cPickle
import pickle
import gzip
import os
import sys
import theano
import theano.tensor as T
import numpy as np
import pdb
import msgpack_numpy

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow)
    shared_y = theano.shared(
            np.asarray(data_y, dtype=theano.config.floatX),
            borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

# def load_data(dataset):
#     f = gzip.open(dataset, 'rb')
#     train_set, valid_set = cPickle.load(f)[0:2]
#     f.close()
#     train_set_x, train_set_y = shared_dataset(train_set)
#     valid_set_x, valid_set_y = shared_dataset(valid_set)
#     return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]

def load_data(dataset):
    f = open(dataset, 'rb')
    train_set, valid_set = pickle.load(f)[0:2]
    f.close()
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]

def get_files(vec_folder):
    file_names = os.listdir(vec_folder)
    file_names.sort()
    if not vec_folder.endswith('/'):
        vec_folder += '/'
    for i in range(len(file_names)):
        file_names[i] = vec_folder + file_names[i]
    return file_names

def load_data_xy(file_names):
    datas  = []
    labels = []
    for file_name in file_names:
        f = open(file_name, 'rb')
        x, y = pickle.load(f)
        datas.append(x)
        labels.append(y)
    combine_d = np.vstack(datas)
    combine_l = np.hstack(labels)
    return combine_d, combine_l

def load_data_split_pickle(dataset):

    valid_folder, train_folder = dataset
    valid_file_names = get_files(valid_folder)
    train_file_names = get_files(train_folder)
    valid_set = load_data_xy(valid_file_names)
    train_set = load_data_xy(train_file_names)

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]


def load_numpy_data(train_image_file, test_image_file):
    train_x, train_y = msgpack_numpy.load(open(train_image_file,'rb'))
    test_x, test_y = msgpack_numpy.load(open(test_image_file,'rb'))

    return train_x, train_y, test_x, test_y
