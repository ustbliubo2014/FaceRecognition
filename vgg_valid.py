# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: vgg_valid.py
@time: 2016/7/18 22:29
"""

import logging
import os
from time import time
if not os.path.exists('log'):
    os.mkdir('log')

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/vgg_valid.log',
                    filemode='w')


import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
from Interface.cluster import imresize, np, imread, cal_distance, load_deepid_model
import msgpack
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pdb
from sklearn.cross_validation import train_test_split
import msgpack_numpy
from sklearn.ensemble import RandomForestClassifier
from CNN_Model.residual import extract_feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from CNN_Model.residual import extract_feature
from sklearn.model_selection import KFold
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB


model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'
pic_shape = (50, 50, 3)

# weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.300.new_shape.rgb.deepid.weight'
# model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.300.new_shape.rgb.deepid.model'
# pic_shape = (78, 62, 3)

model, get_Conv_FeatureMap = load_deepid_model(model_file, weight_file)
# model, get_Conv_FeatureMap = extract_feature(model_file, weight_file)

lfw_folder = '/data/liubo/face/vgg_face_dataset/all_data/pictures_box'
pair_file = '/data/liubo/face/vgg_face_dataset/all_data/vgg_pair.txt'


# 测试lfw的准确率
def cal_two_pic_distance(pic_path1,pic_path2):
    im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(pic_shape[0], pic_shape[1], pic_shape[2])),
                                  (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
    im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(pic_shape[0], pic_shape[1], pic_shape[2])),
                                  (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
    im1 = im1 / 255.0
    im2 = im2 / 255.0

    im1_feature = get_Conv_FeatureMap([im1, 0])[0]
    im2_feature = get_Conv_FeatureMap([im2, 0])[0]
    dist = cal_distance((im1_feature, im2_feature))
    return dist


def main():

    same_dist_list = []
    no_same_dist_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0]
            person_path = os.path.join(lfw_folder, person)
            pic_list = os.listdir(person_path)
            if len(pic_list) == 1:
                print 'error person :', person
                continue
            else:
                np.random.shuffle(pic_list)
                pic_path1 = os.path.join(person_path, pic_list[0])
                pic_path2 = os.path.join(person_path, pic_list[1])
                dist = cal_two_pic_distance(pic_path1, pic_path2)
                same_dist_list.append(dist)
        elif len(tmp) == 4:
            person1 = tmp[0]
            person1_path = os.path.join(lfw_folder, person1)
            pic1_list = os.listdir(person1_path)
            person2 = tmp[2]
            person2_path = os.path.join(lfw_folder, person2)
            pic2_list = os.listdir(person2_path)
            if len(pic1_list) > 0 and len(pic2_list) > 0:
                np.random.shuffle(pic1_list)
                np.random.shuffle(pic2_list)
                pic_path1 = os.path.join(person1_path, pic1_list[0])
                pic_path2 = os.path.join(person2_path, pic2_list[0])
                dist = cal_two_pic_distance(pic_path1, pic_path2)
                no_same_dist_list.append(dist)
    # msgpack.dump((same_dist_list, no_same_dist_list), open('vgg_dist.p','wb'))
    return same_dist_list, no_same_dist_list


def main_feature():
    data = []
    label = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0]
            person_path = os.path.join(lfw_folder, person)
            pic_list = os.listdir(person_path)
            if len(pic_list) < 2:
                print 'error person :', person
                continue
            else:
                np.random.shuffle(pic_list)
                pic_path1 = os.path.join(person_path, pic_list[0])
                pic_path2 = os.path.join(person_path, pic_list[1])
                # print pic_path1, pic_path2
                im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(pic_shape[0], pic_shape[1], pic_shape[2])),
                                  (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
                im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(pic_shape[0], pic_shape[1], pic_shape[2])),
                                  (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
                im1 = im1 / 255.0
                im2 =im2 / 255.0
                im1_feature = get_Conv_FeatureMap([im1, 0])[0]
                im2_feature = get_Conv_FeatureMap([im2, 0])[0]
                this_data = []
                # this_data.append(np.abs(list(im1_feature[0] - im2_feature[0])))
                this_data.append((list(im1_feature[0] - im2_feature[0])))
                data.append(this_data)
                label.append(0)
        elif len(tmp) == 4:
            person1 = tmp[0]
            person1_path = os.path.join(lfw_folder, person1)
            pic1_list = os.listdir(person1_path)
            person2 = tmp[2]
            person2_path = os.path.join(lfw_folder, person2)
            pic2_list = os.listdir(person2_path)
            if len(pic1_list) > 0 and len(pic2_list) > 0:
                np.random.shuffle(pic1_list)
                np.random.shuffle(pic2_list)
                pic_path1 = os.path.join(person1_path, pic1_list[0])
                pic_path2 = os.path.join(person2_path, pic2_list[0])
                # print pic_path1, pic_path2
                im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(pic_shape[0], pic_shape[1], pic_shape[2])),
                                  (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
                im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(pic_shape[0], pic_shape[1], pic_shape[2])),
                                  (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
                im1 = im1 / 255.0
                im2 =im2 / 255.0
                im1_feature = get_Conv_FeatureMap([im1, 0])[0]
                im2_feature = get_Conv_FeatureMap([im2, 0])[0]
                this_data = []
                # this_data.append(np.abs(list(im1_feature[0] - im2_feature[0])))
                this_data.append((list(im1_feature[0] - im2_feature[0])))
                data.append(this_data)
                label.append(1)
    # msgpack_numpy.dump((data, label), open('lfw_data_label.p','w'))
    return data, label


def cal_acc(same_dist_list, no_same_dist_list):
    x = []
    y = []
    for dist in same_dist_list:
        x.append(dist)
        y.append(0)
    for dist in no_same_dist_list:
        x.append(dist)
        y.append(1)
    x = np.reshape(np.asarray(x), (len(x),1))
    y = np.asarray(y)
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.1)
    clf = LinearSVC()
    print len(x), len(y)
    clf.fit(train_x, train_y)
    acc = accuracy_score(valid_y, clf.predict(valid_x))
    print acc
    clf = DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    acc = accuracy_score(valid_y, clf.predict(valid_x))
    print acc


if __name__=='__main__':

    # same_dist_list, no_same_dist_list = main()
    # cal_acc(same_dist_list, no_same_dist_list)

    model_folder = '/data/liubo/face/vgg_face_dataset/model/'
    (data, label) = main_feature()
    data = np.asarray(data)
    data = np.reshape(data, newshape=(data.shape[0], data.shape[2]))
    label = np.asarray(label)
    print data.shape, label.shape
    kf = KFold(n_folds=10)
    all_acc = []
    for k, (train, valid) in enumerate(kf.split(data, label)):
        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]
        clf = RandomForestClassifier(n_estimators=500, n_jobs=15)
        clf.fit(train_data, train_label)
        rf_acc = accuracy_score(valid_label, clf.predict(valid_data))
        all_acc.append(rf_acc)
        train_acc = accuracy_score(train_label, clf.predict(train_data))
        print 'valid_acc :', rf_acc, 'train_acc :', train_acc


