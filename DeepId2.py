# -*-coding:utf-8 -*-
__author__ = 'liubo-it'


import os
import msgpack_numpy
from DeepID.DeepId2.DeepId2 import train_valid_deepid2, extract_feature
import numpy as np
from DeepID.data_process.vgg_face.train_valid_test_split import pack_data
import pdb


def main_orl():
    data_folder = '/data/liubo/face/face_DB/ORL/112x92/'
    X_train, y_train, X_test, y_test = msgpack_numpy.load(open(os.path.join(data_folder,'train_test.p')))
    nb_classes = 40
    print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape
    weight_file = os.path.join(data_folder,'orl.deepid2.weight')
    model_file = os.path.join(data_folder, 'orl.deepid2.model')
    print 'model_file : ', model_file, ' weight_file : ', weight_file
    # train_valid_deepid2(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file)
    extract_feature(X_test, model_file, weight_file)

def main_vgg_face():

    data_folder = '/data/liubo/face/vgg_face_dataset/'

    person_folder = '/data/liubo/face/all_pic_data/annotate'
    pack_file = '/data/liubo/face/vgg_face_dataset/train_valid_data.p'
    X_train, y_train, X_test, y_test = pack_data(person_folder, pack_file, is_test=False)

    nb_classes = len(set(list(y_train)))

    X_train = np.reshape(X_train, newshape=(X_train.shape[0], 3, X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], 3, X_test.shape[1], X_test.shape[2]))
    print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape

    weight_file = os.path.join(os.path.join(data_folder, 'model'), 'annotate.small.rgb.rotate.deepid2.weight')
    model_file = os.path.join(os.path.join(data_folder, 'model'), 'annotate.small.rgb.rotate.deepid2.model')
    print 'model_file : ', model_file, ' weight_file : ', weight_file
    error_valid_sample_file = os.path.join(data_folder,'error_valid_sample.txt')
    error_train_sample_file = os.path.join(data_folder,'error_train_sample.txt')
    train_valid_deepid2(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file,
                       error_train_sample_file, error_valid_sample_file)
    # extract_feature(X_test, model_file, weight_file)

if __name__ == '__main__':
    main_vgg_face()
