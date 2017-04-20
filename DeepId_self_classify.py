# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_self_classify.py
@time: 2016/8/3 19:41
@contact: ustb_liubo@qq.com
@annotation: DeepId_self_classify
"""
import sys
import logging
from logging.config import fileConfig
import os
from DeepId_self_valid import self_feature_pack_file
import msgpack_numpy
import numpy as np
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

self_feature_dic = msgpack_numpy.load(open(self_feature_pack_file, 'rb'))

def load_train_data(train_folder):
    person_list = os.listdir(train_folder)
    all_data = []
    all_label = []
    for person_index, person in enumerate(person_list):
        pic_list = os.listdir(os.path.join(train_folder, person))
        for pic in pic_list:
            feature = self_feature_dic.get(pic)
            all_data.append(np.reshape(feature, feature.size))
            all_label.append(person_index)
    all_data = np.asarray(all_data)
    all_label = np.asarray(all_label)
    print all_data.shape, all_label.shape
    return all_data, all_label


if __name__ == '__main__':
    train_folder = '/data/liubo/face/self_train'
    valid_folder = '/data/liubo/face/self_valid'
    valid_data, valid_label = load_train_data(valid_folder)
    train_data, train_label = load_train_data(train_folder)
    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(train_data, train_label)
    predict_label_prob = clf.predict_proba(valid_data)
    for index in range(2, 1000, 5):
        prob_threshold = index * 1.0 / 1000
        right_num = 0
        find_num = 0
        wrong_num = 0
        for index_sample in range(len(predict_label_prob)):
            start = time()
            if np.max(predict_label_prob[index_sample]) > prob_threshold:
                if np.argmax(predict_label_prob[index_sample]) == valid_label[index_sample]:
                    right_num += 1
                else:
                    wrong_num += 1
                find_num += 1
            end = time()
            # print (end - start)
        print prob_threshold, right_num*1.0/(right_num+wrong_num), find_num*1.0/(len(predict_label_prob))

    # predict_label = clf.predict(valid_data)
    # print accuracy_score(valid_label, predict_label)


