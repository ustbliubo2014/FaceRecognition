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
from Interface.cluster import cal_distance, load_deepid_model
from scipy.misc import imread, imresize
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import msgpack_numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import cPickle
from sklearn.metrics import roc_curve, auc
from CNN_Model.google_net import extract_feature as extract_googleNet_feature
import sklearn.metrics.pairwise as pw
import pdb
from sklearn.model_selection import KFold
import traceback


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='lfw_valid.log',
                    filemode='a+')

# model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
# weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'
# pic_shape = (50, 50, 3)
# model, get_Conv_FeatureMap = load_deepid_model(model_file, weight_file)

model_file = '/data/liubo/face/vgg_face_dataset/model/annotate.all_data.mean.small_shape.rgb.deepid_relu.deep_filters.model'
weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate.all_data.mean.small_shape.rgb.deepid_relu.deep_filters.weight'
pic_shape = (50, 50, 3)
model, get_Conv_FeatureMap = load_deepid_model(model_file, weight_file)


# weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.2622.78-62.rgb.deepid.weight'
# model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.2622.78-62.rgb.deepid.model'
# pic_shape = (78, 62, 3)
# model, get_Conv_FeatureMap = load_deepid_model(model_file, weight_file)



lfw_folder = '/data/liubo/face/lfw_face'
pair_file = '/data/liubo/face/lfw_pair.txt'
feature_pack_file = '/data/liubo/face/lfw_feature.p'
pair_feature_pack_file = '/data/liubo/face/lfw_pair_feature.p'
pair_dist_pack_file = '/data/liubo/face/lfw_pair_dist.p'
error_pair_file = '/data/liubo/face/error_pair.txt'


def read_one_rgb_pic(pic_path, pic_shape):
    im = np.transpose(np.reshape(imresize(imread(pic_path), size=(pic_shape[0], pic_shape[1], pic_shape[2])),
                                  (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
    im = im / 255.0
    return im


def extract(pic_path):
    img = read_one_rgb_pic(pic_path, pic_shape)
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    return feature_vector


def filter_path(this_person_feature_list, index_list):
    # 一张图片里有两个人时,这两个人的图片都删掉
    try:
        tmp_list = []
        for index, e in enumerate(this_person_feature_list):
            feature, path = e
        # Alexandre_Despatie_0001.jpg_face_0.jpg
        # Alexandre_Despatie_0001.jpg_face_1.jpg
        # Alina_Kabaeva_0001.jpg_face_0.jpg
            if 'face_0' not in path:
                tmp_list.append(path[:-5])
        for index, e in enumerate(this_person_feature_list):
            feature, path = e
            for del_path in tmp_list:
                if del_path in path:
                    index_list.remove(index)
                    break
    except:
        traceback.print_exc()



def extract_lfw_feature():
    lfw_feature_dic = {} # {person:[feature1,feature2,...,]}
    person_list = os.listdir(lfw_folder)
    for person_index, person in enumerate(person_list):
        # print person_index, person
        person_path = os.path.join(lfw_folder, person)
        pic_list = os.listdir(person_path)
        this_person_feature_list = []
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            this_feature = extract(pic_path)
            this_person_feature_list.append((this_feature, os.path.join(person, pic)))
        lfw_feature_dic[person] = this_person_feature_list
    msgpack_numpy.dump(lfw_feature_dic, open(feature_pack_file, 'wb'))


def extract_pair_feature():
    lfw_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    data = []
    label = []
    pic_path_list = []
    count = 0
    for line in open(pair_file):
        count += 1
        # print count
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0]    # 取该人的两个特征向量
            this_person_feature_list = lfw_feature_dic.get(person, [])
            index_list = range(len(this_person_feature_list))
            np.random.shuffle(index_list)
            filter_path(this_person_feature_list, index_list)
            if len(index_list) < 2:
                continue
            feature1, path1 = this_person_feature_list[index_list[0]]
            feature2, path2 = this_person_feature_list[index_list[1]]
            feature = np.abs(feature1-feature2)
            label.append(0)
            data.append(feature)
            pic_path_list.append('\t'.join([path1, path2]))
        elif len(tmp) == 4:
            person1 = tmp[0]
            person2 = tmp[2]
            # 每个人分别取一个特征向量
            this_person_feature_list1 = lfw_feature_dic.get(person1, [])
            this_person_feature_list2 = lfw_feature_dic.get(person2, [])
            index_list1 = range(len(this_person_feature_list1))
            index_list2 = range(len(this_person_feature_list2))
            np.random.shuffle(index_list1)
            np.random.shuffle(index_list2)
            filter_path(this_person_feature_list1, index_list1)
            filter_path(this_person_feature_list2, index_list2)
            if len(index_list1) < 1 or len(index_list2) < 1:
                continue
            feature1, path1 = this_person_feature_list1[index_list1[0]]
            feature2, path2 = this_person_feature_list2[index_list2[0]]
            feature = np.abs(feature1-feature2)
            label.append(1)
            data.append(feature)
            pic_path_list.append('\t'.join([path1, path2]))
    msgpack_numpy.dump((data, label, pic_path_list), open(pair_feature_pack_file, 'wb'))



# 测试lfw的准确率
def cal_two_pic_distance(pic_path1, pic_path2):
    im1 = read_one_rgb_pic(pic_path1, pic_shape)
    im2 = read_one_rgb_pic(pic_path2, pic_shape)
    im1_feature = get_Conv_FeatureMap([im1, 0])[0]
    im2_feature = get_Conv_FeatureMap([im2, 0])[0]
    dist = cal_distance((im1_feature, im2_feature))
    return dist


def get_one_person_pictures(person_path):
    # 获取一个人的两张图片
    pic_list = os.listdir(person_path)
    if len(pic_list) < 2:
        print 'error person :', person_path
        return None
    else:
        # np.random.shuffle(pic_list)
        pic_path1 = os.path.join(person_path, pic_list[0])
        pic_path2 = os.path.join(person_path, pic_list[1])
        return pic_path1, pic_path2


def get_two_person_pictures(person1_path, person2_path):
    pic1_list = os.listdir(person1_path)
    pic2_list = os.listdir(person2_path)
    if len(pic1_list) > 0 and len(pic2_list) > 0:
        # np.random.shuffle(pic1_list)
        # np.random.shuffle(pic2_list)
        pic_path1 = os.path.join(person1_path, pic1_list[0])
        pic_path2 = os.path.join(person2_path, pic2_list[0])
        return pic_path1, pic_path2
    else:
        return None


def main():
    same_dist_list = []
    no_same_dist_list = []
    pic_path_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0]
            person_path = os.path.join(lfw_folder, person)
            two_path = get_one_person_pictures(person_path)
            if two_path!= None:
                pic_path1, pic_path2 = two_path
                dist = cal_two_pic_distance(pic_path1, pic_path2)
                same_dist_list.append(dist)
                pic_path_list.append(two_path)
        elif len(tmp) == 4:
            person1_path = os.path.join(lfw_folder, tmp[0])
            person2_path = os.path.join(lfw_folder, tmp[2])
            two_path = get_two_person_pictures(person1_path,person2_path)
            if two_path != None:
                pic_path1, pic_path2 = two_path
                dist = cal_two_pic_distance(pic_path1, pic_path2)
                no_same_dist_list.append(dist)
                pic_path_list.append(two_path)
    msgpack_numpy.dump((same_dist_list, no_same_dist_list), open('lfw_dist.p','wb'))
    return same_dist_list, no_same_dist_list


def get_verification_feature(pic_path1, pic_path2):
    im1 = read_one_rgb_pic(pic_path1, pic_shape)
    im2 = read_one_rgb_pic(pic_path2, pic_shape)
    im1_feature = get_Conv_FeatureMap([im1, 0])[0]
    im2_feature = get_Conv_FeatureMap([im2, 0])[0]
    return np.abs(list(im1_feature[0] - im2_feature[0]))


def main_feature():
    data = []
    label = []
    pic_path_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0]
            person_path = os.path.join(lfw_folder, person)
            two_path = get_one_person_pictures(person_path)
            if two_path!= None:
                data.append(get_verification_feature(two_path[0], two_path[1]))
                label.append(0)
                pic_path_list.append(two_path)
        elif len(tmp) == 4:
            person1_path = os.path.join(lfw_folder, tmp[0])
            person2_path = os.path.join(lfw_folder, tmp[2])
            two_path = get_two_person_pictures(person1_path,person2_path)
            if two_path != None:
                data.append(get_verification_feature(two_path[0], two_path[1]))
                label.append(1)
                pic_path_list.append(two_path)
    msgpack_numpy.dump((data, label, pic_path_list), open('lfw_feature.txt', 'wb'))
    return data, label, pic_path_list


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


def main_distance():
    lfw_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    data = []
    label = []
    pic_path_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0] #取该人的两个特征向量
            this_person_feature_list = lfw_feature_dic.get(person, [])
            index_list = range(len(this_person_feature_list))
            np.random.shuffle(index_list)
            filter_path(this_person_feature_list, index_list)
            if len(index_list) < 2:
                continue
            feature1, path1 = this_person_feature_list[index_list[0]]
            feature2, path2 = this_person_feature_list[index_list[1]]
            predicts = pw.cosine_similarity(feature1, feature2)
            label.append(0)
            data.append(predicts)
            pic_path_list.append('\t'.join([path1, path2]))
        elif len(tmp) == 4:
            person1 = tmp[0]
            person2 = tmp[2]
            # 每个人分别取一个特征向量
            this_person_feature_list1 = lfw_feature_dic.get(person1, [])
            this_person_feature_list2 = lfw_feature_dic.get(person2, [])
            index_list1 = range(len(this_person_feature_list1))
            index_list2 = range(len(this_person_feature_list2))
            np.random.shuffle(index_list1)
            np.random.shuffle(index_list2)
            filter_path(this_person_feature_list1, index_list1)
            filter_path(this_person_feature_list2, index_list2)
            if len(index_list1) < 1 or len(index_list2) < 1:
                continue
            index_list1 = np.arange(len(this_person_feature_list1))
            index_list2 = np.arange(len(this_person_feature_list2))
            np.random.shuffle(index_list1)
            np.random.shuffle(index_list2)
            feature1, path1 = this_person_feature_list1[index_list1[0]]
            feature2, path2 = this_person_feature_list2[index_list2[0]]
            predicts = pw.cosine_similarity(feature1, feature2)
            label.append(1)
            data.append(predicts)
            pic_path_list.append('\t'.join([path1, path2]))
    data = np.asarray(data)
    data = np.reshape(data, newshape=(len(data), 1))
    label = np.asarray(label)
    print data.shape, label.shape

    kf = KFold(n_folds=10)
    all_acc = []
    for k, (train, valid) in enumerate(kf.split(data, label)):
        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        rf_acc = accuracy_score(valid_label, clf.predict(valid_data))
        all_acc.append(rf_acc)
        train_acc = accuracy_score(train_label, clf.predict(train_data))
        print 'valid_acc :', rf_acc, 'train_acc :', train_acc, np.mean(valid_label)
    print np.mean(all_acc)


if __name__=='__main__':
    extract_lfw_feature()
    # extract_pair_feature()
    main_distance()
