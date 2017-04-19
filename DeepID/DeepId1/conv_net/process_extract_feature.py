#-*- coding:utf-8 -*-
__author__ = 'liubo-it'


'''
    将所有的extract_feature合并后降维,训练分类模型
'''


import os
import msgpack_numpy
import pdb
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest


def merge_all_feature(patch_data_folder, extract_feature_file, merged_all_feature_file):
    '''
        一共有60个patch
    :param patch_data_folder: patch处理结果的父目录
    :param extract_feature_file: 提取特征的文件名
    :param merged_all_feature_file: 合并后的特征文件
    :return:
    '''
    patch_num = 60
    first_extract_feature_file = os.path.join(patch_data_folder, str(0), extract_feature_file)
    all_X_train_feature, all_Y_train, all_X_test_feature, all_Y_test = msgpack_numpy.load(open(first_extract_feature_file, 'rb'))
    for patch_id in range(1, patch_num):
        current_extract_feature_file = os.path.join(patch_data_folder, str(patch_id), extract_feature_file)
        if not os.path.exists(current_extract_feature_file):
            print 'no file patch_id', patch_id
            continue
        X_train_feature, Y_train, X_test_feature, Y_test = msgpack_numpy.load(open(current_extract_feature_file, 'rb'))
        if np.all(Y_train == all_Y_train) and np.all(Y_test == all_Y_test):
            all_X_train_feature = np.column_stack((all_X_train_feature, X_train_feature))
            all_X_test_feature = np.column_stack((all_X_test_feature, X_test_feature))
        else:
            print 'not same patch_id', patch_id
    print all_X_train_feature.shape, all_Y_train.shape, all_X_train_feature.shape, all_Y_train.shape
    msgpack_numpy.dump((all_X_train_feature, all_Y_train, all_X_test_feature, all_Y_test),open(merged_all_feature_file,'wb'))


def dimension_reduction(merged_all_feature_file):
    all_X_train_feature, all_Y_train, all_X_test_feature, all_Y_test = \
                        msgpack_numpy.load(open(merged_all_feature_file,'rb'))
    tmp = np.column_stack((all_X_train_feature, all_Y_train))
    np.random.shuffle(tmp)
    all_X_train_feature = tmp[:,:all_X_train_feature.shape[1]]
    all_Y_train = tmp[:,-1]
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    clf.fit(all_X_train_feature, all_Y_train)
    print 'RandomForestClassifier　acc :', accuracy_score(all_Y_test, clf.predict(all_X_test_feature))
    # acc :　 0.997350993377
    selection = SelectKBest(k=200)
    selection.fit(all_X_train_feature, all_Y_train)
    all_X_train_feature_reduce = selection.transform(all_X_train_feature)
    all_X_test_feature_reduce = selection.transform(all_X_test_feature)
    clf = RandomForestClassifier(n_estimators=80,n_jobs=-1)
    clf.fit(all_X_train_feature_reduce, all_Y_train)
    print 'selection RandomForestClassifier　acc :', accuracy_score(all_Y_test, clf.predict(all_X_test_feature_reduce))
    # selection RandomForestClassifier　acc : 0.992494481236


if __name__ == '__main__':
    patch_data_folder = '/home/data/dataset/images/youtube/patch_all_data/'
    extract_feature_file = 'extract_feature.p'
    merged_all_feature_file = '/home/data/dataset/images/youtube/patch_all_data_para/extract_feature.p'
    # merge_all_feature(patch_data_folder, extract_feature_file, merged_all_feature_file)
    dimension_reduction(merged_all_feature_file)
