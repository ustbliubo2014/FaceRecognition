#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: threshold_cal.py
@time: 2016/6/8 10:55
@contact: ustb_liubo@qq.com
@annotation: threshold_cal ： 计算两个阈值(肯定是一个人,肯定不是一个人)[两个都是求最小值]
"""


import os
from scipy.misc import imread, imsave, imresize
import numpy as np
from load_DeepId_model import load_deepid_model
from sklearn.naive_bayes import GaussianNB
from recog_util import cal_distance
import msgpack_numpy
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pdb
import cPickle
from collections import Counter


pic_shape = (50, 50, 3)
deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'



def cal_pic_distance(im_path_list, all_person):
    print 'load deepid model'
    model, get_Conv_FeatureMap = load_deepid_model(deepid_model_file, deepid_weight_file)
    im_feature_list = []
    for index, path in enumerate(im_path_list):
        im = np.transpose(np.reshape(imresize(imread(path), size=(50,50,3)), (1,50,50,3)), (0,3,1,2))
        im_feature = get_Conv_FeatureMap([im, 0])[0]
        im_feature_list.append(im_feature)

    all_score = []
    all_label = []
    # 计算所有dist
    for index_i in range(len(im_feature_list)):
        print index_i
        for index_j in range(0, len(im_feature_list)):
            if index_i == index_j:
                continue
            feature_i = im_feature_list[index_i]
            feature_j = im_feature_list[index_j]
            dist = cal_distance((feature_i, feature_j))
            all_score.append(dist)
            if all_person[index_i] == all_person[index_j]:
                all_label.append(1) # 是同一个人
            else:
                all_label.append(0)
    return all_score, all_label


if __name__ == '__main__':
    # folder = '/data/liubo/face/self'
    # person_list = os.listdir(folder)
    # all_pic_path = []
    # all_person = []
    # for person in person_list:
    #     if person == 'unknown' or person.startswith('new_person'):
    #         continue
    #     person_path = os.path.join(folder, person)
    #     pic_list = os.listdir(person_path)
    #     for pic in pic_list:
    #         pic_path = os.path.join(person_path, pic)
    #         all_pic_path.append(pic_path)
    #         all_person.append(person)
    # all_score, all_label = cal_pic_distance(all_pic_path, all_person)
    # msgpack_numpy.dump((all_score, all_label), open('all_score_label.p','wb'))
    #
    all_score, all_label = msgpack_numpy.load(open('all_score_label.p','rb'))
    count = Counter(all_label)
    print count
    all_score = np.reshape(np.asarray(all_score),(len(all_score), 1))
    all_label = np.asarray(all_label)
    gnb = GaussianNB()
    train_data, test_data, train_label, test_label = train_test_split(all_score, all_label)

    gnb.fit(train_data, train_label)
    gnb.predict_proba(test_data)
    print accuracy_score(test_label, gnb.predict(test_data))
    cPickle.dump(gnb, open('/data/liubo/face/vgg_face_dataset/model/dist_prob.p','wb'))

    pdb.set_trace()