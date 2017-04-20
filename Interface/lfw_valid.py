# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: lfw_valid.py
@time: 2016/7/15 11:50
@contact: ustb_liubo@qq.com
@annotation: lfw_valid
"""
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
from cluster import load_deepid_model, imresize, np, imread, deepid_model_file, deepid_weight_file, cal_distance
import msgpack
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pdb
from sklearn.cross_validation import train_test_split
import msgpack_numpy
from sklearn.ensemble import RandomForestClassifier
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='lfw_valid.log',
                    filemode='a+')

model, get_Conv_FeatureMap = load_deepid_model(deepid_model_file, deepid_weight_file)

# 测试lfw的准确率
def cal_two_pic_distance(pic_path1,pic_path2):
    # im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(156, 124, 3)), (1, 156, 124, 3)), (0, 3, 1, 2))
    # im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(156, 124, 3)), (1, 156, 124, 3)), (0, 3, 1, 2))

    # im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(50, 50, 3)), (1, 50, 50, 3)), (0, 3, 1, 2))
    # im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(50, 50, 3)), (1, 50, 50, 3)), (0, 3, 1, 2))

    im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(78, 62, 3)), (1, 78, 62, 3)), (0, 3, 1, 2))
    im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(78, 62, 3)), (1, 78, 62, 3)), (0, 3, 1, 2))

    im1_feature = get_Conv_FeatureMap([im1, 0])[0]
    im2_feature = get_Conv_FeatureMap([im2, 0])[0]
    dist = cal_distance((im1_feature, im2_feature))
    return dist


def main():
    lfw_folder = '/data/liubo/face/lfw_face'
    pair_file = '/data/liubo/face/lfw_pair.txt'
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
    msgpack.dump((same_dist_list, no_same_dist_list), open('dist.p','wb'))


def main_max_min():
    lfw_folder = '/data/liubo/face/lfw_face'
    pair_file = '/data/liubo/face/lfw_pair.txt'
    same_dist_list = []
    no_same_dist_list = []
    count = 0
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
                this_dist = []
                for index_i in range(len(pic_list[0:10])):
                    for index_j in range(index_i+1, len(pic_list[0:10])):
                        dist = cal_two_pic_distance(
                            os.path.join(person_path, pic_list[index_i]),
                            os.path.join(person_path, pic_list[index_j]))
                        this_dist.append(dist)
                same_dist_list.append(np.min(this_dist))

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

    msgpack_numpy.dump((same_dist_list, no_same_dist_list), open('dist_max_min.p','wb'))


def main_feature():
    lfw_folder = '/data/liubo/face/lfw_face'
    pair_file = '/data/liubo/face/lfw_pair.txt'
    data = []
    label = []
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
                im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(78, 62, 3)), (1, 78, 62, 3)), (0, 3, 1, 2))
                im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(78, 62, 3)), (1, 78, 62, 3)), (0, 3, 1, 2))
                # im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(50, 50, 3)), (1, 50, 50, 3)), (0, 3, 1, 2))
                # im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(50, 50, 3)), (1, 50, 50, 3)), (0, 3, 1, 2))
                im1_feature = get_Conv_FeatureMap([im1, 0])[0]
                im2_feature = get_Conv_FeatureMap([im2, 0])[0]
                this_data = []
                this_data.extend(list(im1_feature[0]))
                this_data.extend(list(im2_feature[0]))
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
                im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(78, 62, 3)), (1, 78, 62, 3)), (0, 3, 1, 2))
                im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(78, 62, 3)), (1, 78, 62, 3)), (0, 3, 1, 2))
                # im1 = np.transpose(np.reshape(imresize(imread(pic_path1), size=(50, 50, 3)), (1, 50, 50, 3)), (0, 3, 1, 2))
                # im2 = np.transpose(np.reshape(imresize(imread(pic_path2), size=(50, 50, 3)), (1, 50, 50, 3)), (0, 3, 1, 2))
                im1_feature = get_Conv_FeatureMap([im1, 0])[0]
                im2_feature = get_Conv_FeatureMap([im2, 0])[0]
                this_data = []
                this_data.extend(list(im1_feature[0]))
                this_data.extend(list(im2_feature[0]))
                data.append(this_data)
                label.append(1)
    msgpack_numpy.dump((data, label), open('lfw_data_label.p','w'))
    # return data, label





def cal_acc(dist_file):
    x = []
    y = []
    same_dist_list, no_same_dist_list = msgpack.load(open(dist_file, 'rb'))
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
    # main_feature()
    # main()
    # cal_acc('dist.p')
    # main_max_min()
    # cal_acc('dist_max_min.p')

    (data, label) = msgpack_numpy.load(open('lfw_data_label.p','r'))
    data = np.asarray(data)
    label = np.asarray(label)
    train_x, valid_x, train_label, valid_label = train_test_split(data, label, test_size=0.1)
    print train_x.shape, valid_x.shape, train_label.shape, valid_label.shape
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=15)
    clf.fit(train_x, train_label)
    acc = accuracy_score(valid_label, clf.predict(valid_x))
    train_acc = accuracy_score(train_label, clf.predict(train_x))
    print acc, train_acc
