# -*- coding:utf-8 -*-
__author__ = 'liubo-it'


from scipy.misc import imread
import os
import numpy as np
import msgpack_numpy

def load_data(folder):
    file_list = os.listdir(folder)
    np.random.shuffle(file_list)
    all_data = []
    all_label = []
    for file_name in file_list:
        label = int(file_name.split('_')[1]) - 1
        absolute_path = os.path.join(folder, file_name)
        arr = np.transpose(imread(absolute_path),(2,0,1))
        all_data.append(arr)
        all_label.append(label)
    all_data = np.asarray(all_data)
    all_label = np.asarray(all_label)
    return all_data, all_label


if __name__ == '__main__':
    train_folder = '/data/liubo/face/face_DB/ORL/train/'
    test_folder = '/data/liubo/face/face_DB/ORL/test/'
    msgpack_data_file = '/data/liubo/face/face_DB/ORL/train_test.p'
    train_data, train_label = load_data(train_folder)
    test_data, test_label = load_data(test_folder)
    msgpack_numpy.dump((train_data, train_label, test_data, test_label), open(msgpack_data_file,'wb'))
    print train_data.shape, train_label.shape, test_data.shape, test_label.shape

