# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_self_valid.py
@time: 2016/8/3 17:40
@contact: ustb_liubo@qq.com
@annotation: DeepId_self_valid
"""
import sys
import logging
from logging.config import fileConfig
import os
from Interface.cluster import cal_distance, load_deepid_model
from scipy.misc import imread, imsave, imresize
from random import randint
import pdb
import msgpack
import msgpack_numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sklearn.metrics.pairwise as pw
from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import KFold
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

model_file = '/data/liubo/face/vgg_face_dataset/model/annotate.all_data.mean.small.rgb.deepid_relu.small_filters.model'
weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate.all_data.mean.small.rgb.deepid_relu.small_filters.weight'
pic_shape = (78, 62, 3)
model, get_Conv_FeatureMap = load_deepid_model(model_file, weight_file)
same_person_id = 0
no_same_person_id = 1
self_folder = '/data/liubo/face/self'
self_feature_pack_file = '/data/liubo/face/self_deepid_feature.p'


def read_one_rgb_pic(pic_path, pic_shape):
    im = imresize(imread(pic_path), size=(pic_shape[0], pic_shape[1], pic_shape[2]))
    channel_mean = np.array([ 143.69581142,  113.83085749,  100.20530457])
    im = im - channel_mean
    im = np.transpose(np.reshape(im, (1, pic_shape[0], pic_shape[1], pic_shape[2])), (0, 3, 1, 2))
    im = im / 255.0
    return im


def extract(pic_path):
    img = read_one_rgb_pic(pic_path, pic_shape)
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    return feature_vector


def extract_self_data_feature():
    self_feature_dic = {}    # {pic_path:feature}
    person_list = os.listdir(self_folder)
    for person_index, person in enumerate(person_list):
        print person_index, person
        person_path = os.path.join(self_folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            this_feature = extract(pic_path)
            self_feature_dic[pic] = this_feature
    msgpack_numpy.dump(self_feature_dic, open(self_feature_pack_file, 'wb'))


def load_pair_data():
    folder = '/data/liubo/face/self'
    person_list = os.listdir(folder)
    person_path_dic = {}   # {person:[pic_path]}
    person_num = len(person_list)
    for person in person_list:
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        person_path_dic[person] = pic_list
    pair_list = []
    for person in person_path_dic:
        this_person_path_list = person_path_dic.get(person)
        path_num = len(this_person_path_list)
        sample_num = path_num * (path_num - 1) / 2
        count = 0
        while count < sample_num:
            other_person = person_list[randint(0, person_num-1)]
            if other_person == person:
                continue
            count += 1
            other_person_path = person_path_dic.get(other_person)
            pair_list.append((
                this_person_path_list[randint(0, path_num-1)],
                other_person_path[randint(0, len(other_person_path)-1)],
                no_same_person_id
                ))
        for index_i in range(path_num):
            for index_j in range(index_i+1, path_num):
                pair_list.append((
                    this_person_path_list[index_i],
                    this_person_path_list[index_j],
                    same_person_id
                ))
    return pair_list


def main_distance():
    self_feature_dic = msgpack_numpy.load(open(self_feature_pack_file, 'rb'))
    data = []
    label = []
    path_pair_list = load_pair_data()
    for pic_path1, pic_path2, this_label in path_pair_list:
        feature1 = self_feature_dic.get(pic_path1)
        feature2 = self_feature_dic.get(pic_path2)
        if feature1 == None or feature2 == None:
            pdb.set_trace()

        predicts = pw.cosine_similarity(feature1, feature2)
        label.append(this_label)
        data.append(predicts)

    data = np.asarray(data)
    data = np.reshape(data, newshape=(len(data), 1))
    label = np.asarray(label)
    print data.shape, label.shape

    all_acc = []
    for k in range(10):
        train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.1)
        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        all_acc.append(acc)
        print acc
        # cPickle.dump(clf, open('/data/liubo/face/vgg_face_dataset/model/verification_model', 'wb'))
    print 'mean acc', np.mean(all_acc)


if __name__ == '__main__':
    extract_self_data_feature()
    main_distance()

