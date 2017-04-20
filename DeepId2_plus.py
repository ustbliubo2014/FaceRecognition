# -*-coding:utf-8 -*-
__author__ = 'liubo-it'


import os
import msgpack_numpy
from DeepID.DeepId2.DeepId2_plus import train_valid_deepid2plus, extract_feature


if __name__ == '__main__':
    data_folder = '/data/liubo/face/face_DB/ORL/112x92/'
    X_train, y_train, X_test, y_test = msgpack_numpy.load(open(os.path.join(data_folder,'train_test.p')))
    nb_classes = 40
    print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape
    weight_file = os.path.join(data_folder,'orl.deepid2plus.weight')
    model_file = os.path.join(data_folder, 'orl.deepid2plus.model')
    print 'model_file : ', model_file, ' weight_file : ', weight_file
    train_valid_deepid2plus(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file)
    # extract_feature(X_test, model_file, weight_file)
