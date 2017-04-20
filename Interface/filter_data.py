#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: filter_data.py
@time: 2016/6/1 17:42
@contact: ustb_liubo@qq.com
@annotation: filter_data : 一个人的数据比较多,通过聚类删掉一些
"""

from cluster import load_unknown_data
import shutil
import os
from sklearn.cluster import AgglomerativeClustering
from load_DeepId_model import load_deepid_model
import numpy as np
from scipy.misc import imsave
from recog_util import cal_distance
import pdb
import sys
from sklearn.neighbors import LSHForest
from time import time


deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'


def find_k_neighbors_with_lsh(self, one_pic_feature):
    try:
        tmp = self.lshf.kneighbors(one_pic_feature, n_neighbors=self.n_neighbors, return_distance=True)
        result_label = np.asarray(self.all_labels)[tmp[1][0]]
        return zip(tmp[0][0], result_label)
    except:
        return None

def filter_data(folder, same_dist_threshold):
    lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=10)
    data, pic_list = load_unknown_data(folder)
    for index in range(len(pic_list)):
        pic_list[index] = os.path.join(folder, pic_list[index])
    new_data = np.transpose(data, (0,3,1,2))
    # print 'load deepid model'
    # model, get_Conv_FeatureMap = load_deepid_model(deepid_model_file, deepid_weight_file)
    start = time()
    data_feature = get_Conv_FeatureMap([new_data,0])[0]
    # print 'data_feature.shape :', data_feature.shape
    # print 'pic_list :', pic_list

    lshf.fit(data_feature, pic_list)
    need_remove_list = set()
    no_same_feature_list = []
    for index_i in range(len(data_feature)):
        if len(no_same_feature_list) == 0:
            no_same_feature_list.append(data_feature[index_i:index_i+1])
        else:
            tmp = lshf.kneighbors(data_feature[index_i:index_i+1], n_neighbors=10, return_distance=True)
            tmp = zip(tmp[0][0], tmp[1][0])
            for index_j in range(len(tmp)):
                if tmp[index_j][1] == index_i:
                    continue
                if tmp[index_j][0] < same_dist_threshold:
                    need_remove_list.add(pic_list[index_i])
    for path in need_remove_list:
        try:
            os.remove(path)
        except:
            print 'error path :', path
            continue

    end = time()
    print 'filter time :', (end - start)
    return len(no_same_feature_list)


if __name__ == '__main__':
    father_folder = '/data/liubo/face/self/'
    person_list = os.listdir(father_folder)
    model, get_Conv_FeatureMap = load_deepid_model(deepid_model_file, deepid_weight_file)
    for person in person_list:
        print person
        # if not (person == 'unknown' or 'Must_Same' in person or 'Maybe_same' in person):
        #     continue
        # if person != sys.argv[1]:
        #     continue
        person_folder = os.path.join(father_folder,person)
        no_same_num = filter_data(person_folder, same_dist_threshold=8)
        print person, no_same_num

