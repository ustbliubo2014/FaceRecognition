# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: research_model_test.py
@time: 2016/8/29 14:21
@contact: ustb_liubo@qq.com
@annotation: research_model_test
"""
import sys
import logging
from logging.config import fileConfig
import os
import urllib2
import urllib
from time import time
import traceback
import pdb
import msgpack
import numpy as np
import msgpack_numpy
import sklearn.metrics.pairwise as pw
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import cPickle

reload(sys)
sys.setdefaultencoding("utf-8")


# curl --data-binary @liubo-it1468381751.27.png_face_0.jpg "10.16.28.15:8001/test.html"
pair_file = '/data/liubo/face/self_all_pair.txt'
feature_pack_file = '/data/liubo/face/research_self_feature.p'


def valid_one_pic_recognize(pic_path):
    url = "10.16.28.15:8001/test.html"
    command = 'curl --data-binary @%s %s > tmp.txt'%(pic_path, url)
    os.system(command)
    feature = open('tmp.txt').read().rstrip().split(',')[:-1]
    feature = map(float, feature)
    return feature


def extract_feature():
    root_dir = 'tmp'
    person_list = os.listdir(root_dir)
    path_feature_dic = {}
    for person in person_list:
        person_path = os.path.join(root_dir, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            feature = valid_one_pic_recognize(pic_path)
            path_feature_dic[pic] = feature
    msgpack.dump(path_feature_dic, open('research_feature.p', 'w'))


def main_distance():
    all_data = []
    all_label = []
    all_pic_path_list = []
    count = 0
    path_feature_dic = msgpack.load(open('research_feature.p', 'r'))
    not_in = 0
    not_in_pair = {}
    for line in open(pair_file):
        if count % 100 == 0:
            print count
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            path1 = os.path.split(tmp[0])[1]
            path2 = os.path.split(tmp[1])[1]
            label = int(tmp[2])
            if path1 in path_feature_dic and path2 in path_feature_dic:
                try:
                    feature1 = np.asarray(path_feature_dic.get(path1))
                    feature2 = np.asarray(path_feature_dic.get(path2))
                    if len(feature1) < 100 or len(feature2) < 100:
                        print path1, path2
                        not_in += 1
                        not_in_pair[(path1, path2)] = 1
                        continue
                    feature1 = np.reshape(feature1, newshape=(1, feature1.shape[0]))
                    feature2 = np.reshape(feature2, newshape=(1, feature2.shape[0]))
                    predicts = pw.cosine_similarity(feature1, feature2)
                    all_data.append(predicts)
                    all_label.append(label)
                    all_pic_path_list.append((path1, path2))
                except:
                    pdb.set_trace()
            else:
                pdb.set_trace()
    print not_in
    msgpack_numpy.dump((all_data, all_label, all_pic_path_list), open(feature_pack_file, 'wb'))
    cPickle.dump(not_in_pair, open('not_in_pair.p', 'w'))


if __name__ == '__main__':
    # extract_feature()
    main_distance()

    (all_data, all_label, all_pic_path_list) = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    all_data = np.asarray(all_data)
    all_data = np.reshape(all_data, newshape=(all_data.shape[0], all_data.shape[2]))
    all_label = np.asarray(all_label)
    all_pic_path_list = np.asarray(all_pic_path_list)
    print all_data.shape, all_label.shape

    all_acc = []

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
    clf = LinearSVC()
    clf.fit(all_data, all_label)
    cPickle.dump(clf, open('/data/liubo/face/vgg_face_model/research_verification_model', 'wb'))
