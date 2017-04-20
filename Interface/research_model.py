# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: research_model.py : 研究院的模型
@time: 2016/11/4 22:59
"""

import sys
import logging
from logging.config import fileConfig
import os
from scipy.misc import imsave
import traceback
import cPickle
from time import time
import requests
import numpy as np
import pdb
import cv2
import msgpack_numpy
import msgpack
import sklearn.metrics.pairwise as pw
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
import shutil

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


FEATURE_DIM = 256
same_pic_threshold = 0.90
upper_verif_threshold = 0.80
lower_verif_threshold = 0.65
port = 6666
pair_file = 'self_all_pair.txt'
feature_pack_file = 'research_feature_pack.p'
verification_model_file = '/data/liubo/face/vgg_face_model/research_verification_model'
verification_model = cPickle.load(open(verification_model_file, 'rb'))
nearest_num = 5   # 计算当前图片与以前5张图片的相似度
nearest_time_threshold = 30
verification_same_person = 0


def create_train_valid_data(folder='/data/liubo/face/research_feature_self'):
    # 根据已经存在的数据训练人脸验证模型
    person_list = os.listdir(folder)
    path_feature_dic = {}  #
    for person in person_list:
        person_path = os.path.join(folder, person)
        pic_feature_list = os.listdir(person_path)
        for pic_feature_path in pic_feature_list:
            pic_feature_path = os.path.join(person_path, pic_feature_path)
            pic_feature = msgpack_numpy.load(open(pic_feature_path, 'rb'))
            path_feature_dic[pic_feature_path] = pic_feature
    msgpack.dump(path_feature_dic, open('research_feature.p', 'wb'))


def train_valid_verif_model():
    all_data = []
    all_label = []
    all_pic_path_list = []
    count = 0
    path_feature_dic = msgpack.load(open('research_feature.p', 'rb'))
    not_in = 0
    not_in_pair = {}
    for line in open(pair_file):
        if count % 100 == 0:
            print count
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            path1 = tmp[0]
            path2 = tmp[1]
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
                    traceback.print_exc()
                    # pdb.set_trace()
            else:
                traceback.print_exc()
                # pdb.set_trace()
    msgpack_numpy.dump((all_data, all_label, all_pic_path_list), open(feature_pack_file, 'wb'))

    (all_data, all_label, all_pic_path_list) = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    all_data = np.asarray(all_data)
    all_data = np.reshape(all_data, newshape=(all_data.shape[0], all_data.shape[2]))
    all_label = np.asarray(all_label)
    print all_data.shape, all_label.shape


    kf = KFold(len(all_label), n_folds=10)
    all_acc = []
    for (train, valid) in kf:
        train_data = all_data[train]
        valid_data = all_data[valid]
        train_label = all_label[train]
        valid_label = all_label[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        roc_auc = roc_auc_score(valid_label, clf.predict(valid_data))
        all_acc.append(acc)
        print acc, roc_auc
    print 'mean_acc :', np.mean(all_acc)
    clf = LinearSVC()
    clf.fit(all_data, all_label)
    cPickle.dump(clf, open(verification_model_file, 'wb'))


def extract_feature_from_file(pic_path):
    # curl --data-binary @1478257275.37.jpg "olgpu10.ai.shbt.qihoo.net:8001/test.html"
    try:
        pic_binary_data = open(pic_path, 'rb').read()
        result = extract_feature_from_binary_data(pic_binary_data)
        if result == None:
            return None
        else:
            face_num, frame, feature = result
            return feature
    except:
        return None


def extract_feature_from_numpy(img):
    tmp_file = '/tmp/numpy_img.png'
    imsave(tmp_file, img)
    face_num, frame, feature = extract_feature_from_file(tmp_file)
    return feature


def extract_feature_from_binary_data(pic_binary_data):
    # curl --data-binary @1478257275.37.jpg "olgpu10.ai.shbt.qihoo.net:8001/test.html"
    try:
        start = time()
        request = requests.post("http://olgpu10.ai.shbt.qihoo.net:8001/test.html", pic_binary_data)
        end = time()
        # print request.status_code, (end - start)
        if request.status_code == 200:
            content = request.content
            tmp = content.split('\n')
            if len(tmp) < 3:
                return None
            face_num = int(tmp[0].split(':')[1])
            all_frames = []
            all_feature = []
            for k in range(face_num):
                frame = map(float, tmp[2*k+1].split(','))
                feature = map(float, tmp[2*k+2].split(',')[:-1])
                all_frames.append(frame)
                all_feature.append(feature)
            return face_num, all_frames, all_feature
    except:
        traceback.print_exc()
        return None


def find_big_face(all_frame):
    # 返回面积最大的frame的id
    max_index = 0
    max_size = 0
    for index in range(len(all_frame)):
        this_frame = all_frame[index]
        this_size = this_frame[2] * this_frame[3]
        if this_size > max_size:
            max_size = this_size
            max_index = index
    return max_index


def get_all_img_feature():
    folder = '/tmp/annotate'
    result_pic_folder = '/data/liubo/face/research_self'
    result_feature_folder = '/data/liubo/face/research_feature_self'
    person_list = os.listdir(folder)
    for person in person_list:
        person_path = os.path.join(folder, person)
        result_person_pic_folder = os.path.join(result_pic_folder, person)
        if not os.path.exists(result_person_pic_folder):
            os.makedirs(result_person_pic_folder)
        person_feature_folder = os.path.join(result_feature_folder, person)
        if not os.path.exists(person_feature_folder):
            os.makedirs(person_feature_folder)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            try:
                pic_path = os.path.join(person_path, pic)
                feature = np.asarray(extract_feature_from_file(pic_path)[0])
                shutil.copy(pic_path, os.path.join(result_person_pic_folder, pic))
                msgpack_numpy.dump(feature, open(os.path.join(person_feature_folder, pic + '.p'), 'wb'))
            except:
                traceback.print_exc()
                continue


if __name__ == '__main__':
    pass

    # pic_path = sys.argv[1]
    # feature1 = extract_feature_from_file(pic_path)
    # pic_path = sys.argv[2]
    # feature2 = extract_feature_from_file(pic_path)
    # print cosine_similarity(feature1[0], feature2[0])

    folder = '/home/liubo-it/FaceRecognization/Interface/face_sim_test'
    pic_list = map(lambda x:os.path.join(folder, x), os.listdir(folder))
    for index_i in range(len(pic_list)):
        for index_j in range(index_i+1, len(pic_list)):
            pic_path1 = pic_list[index_i]
            pic_path2 = pic_list[index_j]
            feature1 = np.asarray(extract_feature_from_file(pic_path1))
            feature1 = np.reshape(feature1, (1, feature1.size))
            feature2 = np.asarray(extract_feature_from_file(pic_path2))
            feature2 = np.reshape(feature2, (1, feature2.size))
            print pic_path1, pic_path2, cosine_similarity(feature1[0], feature2[0])
