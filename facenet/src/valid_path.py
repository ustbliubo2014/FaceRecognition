# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: valid_path.py
@time: 2017/2/7 9:43
@contact: ustb_liubo@qq.com
@annotation: valid_path
"""
import sys
import logging
from logging.config import fileConfig
import os
import msgpack_numpy
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    (paths, emb_array, actual_issame) = msgpack_numpy.load(open('lfw_feature.p', 'rb'))
    data = []
    pair_paths = []
    for index in range(len(actual_issame)):
        data.append(cosine_similarity(emb_array[2*index:2*index+1], emb_array[2*index+1:2*index+2])[0][0])
        pair_paths.append(str(paths[2*index]) + '\t' + str(paths[2*index+1]))
    data = np.reshape(np.array(data), (len(data), 1))
    label = np.reshape(np.array(actual_issame), (len(actual_issame), 1))
    pair_paths = np.array(pair_paths)

    kf = KFold(len(label), n_folds=10)
    all_acc = []
    f = open('error.txt', 'w')
    for (train, valid) in kf:
        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]
        train_path = pair_paths[train]
        valid_path = pair_paths[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        roc_auc = roc_auc_score(valid_label, clf.predict(valid_data))
        for index in range(len(valid_data)):
            if valid_label[index] != clf.predict(np.reshape(valid_data[index], (1, 1))):
                f.write(str(index) + '\t' + valid_path[index] + '\n')
        all_acc.append(acc)
        print acc, roc_auc
    f.close()
    all_acc.sort(reverse=True)
    print 'mean_acc :', np.mean(all_acc[:])

