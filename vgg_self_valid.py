# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: vgg_self_valid.py
@time: 2016/8/3 16:36
@contact: ustb_liubo@qq.com
@annotation: vgg_self_valid : 用测试lfw的方法测试自己采的数据集
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
import caffe
from scipy.misc import imread, imsave, imresize
from random import randint
import pdb
import msgpack
import msgpack_numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sklearn.metrics.pairwise as pw
from sklearn.cross_validation import train_test_split
import cPickle
from sklearn.model_selection import KFold

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

avg = np.array([129.1863, 104.7624, 93.5940])
pic_shape = (224, 224, 3)
same_person_id = 0
no_same_person_id = 1
layer = 'fc7'
self_folder = '/data/liubo/face/tmp'
self_feature_pack_file = '/data/liubo/face/self_vgg_feature.p'
error_pair_file = '/data/liubo/face/self_vgg_error.txt'
all_pair_file = '/data/liubo/face/self_vgg_all_pair_score.txt'

caffe.set_mode_gpu()
caffe.set_device(3)

net = caffe.Net("/home/liubo-it/VGGFaceModel-master/VGG_FACE_deploy.prototxt",
                "/home/liubo-it/VGGFaceModel-master/VGG_FACE.caffemodel",
                caffe.TEST)
verification_model_file = '/data/liubo/face/vgg_face_dataset/model/verification_model'
verification_model = cPickle.load(open(verification_model_file, 'rb'))
#
def read_one_rgb_pic(pic_path, pic_shape=(224, 224, 3)):

    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def extract(pic_path):
    img = read_one_rgb_pic(pic_path, pic_shape)
    net.blobs['data'].data[...] = img
    net.forward(data=img)
    feature_vector = net.blobs[layer].data[0].copy()
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
    folder = self_folder
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
        feature1 = np.reshape(feature1, newshape=(1, feature1.shape[0]))
        feature2 = np.reshape(feature2, newshape=(1, feature2.shape[0]))
        predicts = pw.cosine_similarity(feature1, feature2)
        label.append(this_label)
        data.append(predicts)

    data = np.asarray(data)
    data = np.reshape(data, newshape=(len(data), 1))
    label = np.asarray(label)
    print data.shape, label.shape

    all_acc = []
    f = open(error_pair_file, 'w')
    kf = KFold(n_folds=10)
    # for k in range(10):
    path_pair_list = np.asarray(path_pair_list)
    for train_index, test_index in kf.split(path_pair_list):
        train_data, valid_data = data[train_index], data[test_index]
        train_label, valid_label = label[train_index], label[test_index]
        train_path_list, valid_path_list = path_pair_list[train_index], path_pair_list[test_index]
        # train_data, valid_data, train_label, valid_label, train_path_list, valid_path_list \
        #             = train_test_split(data, label, path_pair_list, test_size=0.1)
        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        # acc = accuracy_score(valid_label, verification_model.predict(valid_data))
        for index in range(len(valid_label)):
            if valid_label[index] != clf.predict(valid_data[index:index+1]):
                f.write(str(valid_path_list[index][0])+'\t'+
                        str(valid_path_list[index][1])+'\t'+str(valid_path_list[index][2]+
                                                                '\t'+str(valid_data[index][0]))+'\n')
        all_acc.append(acc)
        print 'acc :', acc
    print 'mean acc', np.mean(all_acc)
    f.close()
    # 找上限
    upper_cos_threshold = float(sys.argv[1])
    lower_cos_threshold = float(sys.argv[2])
    right = wrong = no_find = 0.0
    for index in range(len(label)):
        if data[index] > upper_cos_threshold:
            if label[index] == same_person_id:
                right += 1
            else:
                wrong += 1
        elif data[index] < lower_cos_threshold:
            if label[index] != same_person_id:
                right += 1
            else:
                wrong += 1
        else:
            no_find += 1
        # print right, wrong, no_find
    print 'acc :', right*1.0/(right+wrong), 'recall :', (right+wrong)/(right+wrong+no_find)
    # clf = LinearSVC()
    # clf.fit(data, label)
    # print 'train acc:', accuracy_score(label, clf.predict(data))
    # cPickle.dump(clf, open('/data/liubo/face/vgg_face_dataset/model/self_verification_model', 'wb'))
    # f.close()
    # f = open(all_pair_file, 'w')
    # for index in range(len(label)):
            # f.write(str(path_pair_list[index][0])+'\t'+
            #         str(path_pair_list[index][1])+'\t'+str(path_pair_list[index][2]+
            #                                                 '\t'+str(data[index][0]))+'\n')



if __name__ == '__main__':
    extract_self_data_feature()
    main_distance()
