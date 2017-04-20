# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: beiyou_self_valid.py
@time: 2016/8/13 20:24
"""

import logging
import os
import sys
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pdb

def load_data(file_name):
    beiyou_score_list = []
    deepface_score_list = []
    label = []
    path_list = []
    for line in open(file_name):
        if line.startswith('path'):
            continue
        tmp = line.rstrip().split()
        if len(tmp) == 5:
            beiyou_score_list.append(float(tmp[2]))
            label.append(float(tmp[3]))
            deepface_score_list.append(float(tmp[4]))
            path_list.append((tmp[0],tmp[1]))
    beiyou_score_list = np.asarray(beiyou_score_list)
    deepface_score_list = np.asarray(deepface_score_list)
    label = np.asarray(label)
    path_list = np.asarray(path_list)
    return beiyou_score_list, deepface_score_list, label, path_list


def valid(all_data, all_label, all_pic_path_list):
    all_data = np.reshape(all_data, newshape=(all_data.shape[0], 1))
    kf = KFold(n_folds=10)
    all_acc = []
    for k, (train, valid) in enumerate(kf.split(all_data, all_label, all_pic_path_list)):
        train_data = all_data[train]
        valid_data = all_data[valid]
        train_label = all_label[train]
        valid_label = all_label[valid]
        train_path_list = all_pic_path_list[train]
        valid_path_list = all_pic_path_list[valid]
        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        all_acc.append(acc)
        print acc
    print 'mean_acc :', np.mean(all_acc)


def train_all(all_data, all_label, all_pic_path_list, error_file):
    all_data = np.reshape(all_data, newshape=(all_data.shape[0], 1))
    clf = LinearSVC()
    clf.fit(all_data, all_label)
    f = open(error_file, 'w')
    predict_label = clf.predict(all_data)
    pdb.set_trace()
    count = 0
    drop_num = 0
    for index in range(len(all_label)):
        # if all_data[index] < 0.45 and all_data[index] > 0.4:
        #     drop_num += 1
        #     continue
        if all_label[index] != predict_label[index]:
            count += 1
            f.write(all_pic_path_list[index][0]+'\t'+all_pic_path_list[index][1]+'\t'+str(all_data[index][0])+'\n')
    print 1 - count * 1.0 / len(all_data), 1 - drop_num * 1.0 / len(all_data)
    f.close()

if __name__ == '__main__':
    file_name = 'face_pair_score.txt'
    beiyou_score_list, deepface_score_list, label, path_list = load_data(file_name)
    # valid(beiyou_score_list, label, path_list)
    # valid(deepface_score_list, label, path_list)
    train_all(deepface_score_list, label, path_list, 'deepface_error.txt')
    train_all(beiyou_score_list, label, path_list, 'beiyou_error.txt')