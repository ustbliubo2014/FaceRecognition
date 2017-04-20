# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: extract_feature.py
@time: 2016/8/8 15:45
@contact: ustb_liubo@qq.com
@annotation: extract_annotate_feature
"""
import sys
import logging
from logging.config import fileConfig
import os
from lfw_keras_vgg_valid import extract
from util.load_data import load_one_deep_path, load_two_deep_path
import numpy as np
import pdb
import msgpack_numpy
import traceback
from sklearn.cross_validation import train_test_split

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

feature_folder = '/data/pictures_annotate_feature'
data_folder = '/data/pictures_annotate'


def load_train_data(folder):
    data = []
    label = []
    person_path_dic = load_two_deep_path(folder)
    current_id = 0
    for person in person_path_dic:
        try:
            pic_path_list = person_path_dic.get(person)
            for pic_path in pic_path_list:
                pic_feature = extract(pic_path)[0]
                data.append(pic_feature)
                label.append(current_id)
            current_id += 1
            print current_id, person.decode('gbk')
        except:
            print 'error person :', person
            traceback.print_exc()
            continue
    data = np.asarray(data)
    label = np.asarray(label)
    print data.shape, label.shape
    return data, label


def create_train_valid_list(folder):
    sample_list = []
    person_path_dic = load_two_deep_path(folder)
    current_id = 0
    for person in person_path_dic:
        try:
            pic_path_list = person_path_dic.get(person)
            for pic_path in pic_path_list[0:]:
                sample_list.append((pic_path, current_id))
            current_id += 1
            print current_id, person.decode('gbk')
        except:
            print 'error person :', person
            traceback.print_exc()
            continue
    sample_list = np.asarray(sample_list)
    train_sample_list, valid_sample_list = train_test_split(sample_list, test_size=0.1)
    return train_sample_list, valid_sample_list


def extract_person_feature(folder):
    person_feature_list_dic = {}
    person_path_dic = load_two_deep_path(folder)
    current_id = 0
    for person in person_path_dic:
        try:
            pic_path_list = person_path_dic.get(person)
            feature_list = []
            for pic_path in pic_path_list:
                pic_feature = extract(pic_path)
                feature_list.append(pic_feature)
            current_id += 1
            person_feature_list_dic[person] = feature_list
            print current_id, person.decode('gbk')
        except:
            print 'error person :', person
            traceback.print_exc()
            continue
    return person_feature_list_dic


def save_annotate_data():
    data, label = load_train_data(data_folder)
    msgpack_numpy.dump((data, label), open('/data/pictures_annotate_feature/annotate_data.p', 'wb'))


def save_annotate_person_feature():
    person_feature_list_dic = extract_person_feature(data_folder)
    msgpack_numpy.dump(person_feature_list_dic, open('/data/pictures_annotate_feature/person_feature_list_dic.p', 'wb'))


def merge_data():
    (annotate_data, annotate_label) = msgpack_numpy.load(open('/data/pictures_annotate_feature/more_person_data_label.p', 'rb'))
    (self_data, self_label) = msgpack_numpy.load(open('/data/pictures_annotate_feature/self_data_label.p', 'rb'))
    annotate_data = list(annotate_data)
    annotate_label = list(annotate_label)
    start_index = max(set(annotate_label)) + 1
    self_data = list(self_data)
    self_label = list(self_label)
    self_label = map(lambda x: x+start_index, self_label)
    annotate_data.extend(self_data)
    annotate_label.extend(self_label)
    annotate_data = np.asarray(annotate_data)
    annotate_label = np.asarray(annotate_label)
    msgpack_numpy.dump((annotate_data, annotate_label), open('/data/pictures_annotate_feature/more_person_with_self_data_label.p', 'wb'))


def extract_vgg_feature(vgg_folder, pic_num_threshold=50):
    # vgg的数据集,每人的图片排序后取前50张图片(前面的图片比较准确)
    data = []
    label = []
    person_path_dic = load_one_deep_path(vgg_folder)
    current_id = 0
    for person in person_path_dic:
        try:
            pic_path_list = person_path_dic.get(person)
            pic_path_list.sort()
            for pic_path in pic_path_list[:pic_num_threshold]:
                pic_feature = extract(pic_path)
                data.append(pic_feature)
                label.append(current_id)
            current_id += 1
            print current_id, person
        except:
            print 'error person :', person
            traceback.print_exc()
            continue
    data = np.asarray(data)
    label = np.asarray(label)
    return data, label


def save_vgg_data_feature():
    vgg_folder = '/data/liubo/face/vgg_face_dataset/all_data/pictures_box'
    data, label = extract_vgg_feature(vgg_folder)
    msgpack_numpy.dump((data, label), open('/data/pictures_annotate_feature/vgg_data_label.p', 'wb'))


if __name__ == '__main__':
    # train_sample_list, valid_sample_list = create_train_valid_list(data_folder)
    # msgpack_numpy.dump((train_sample_list, valid_sample_list), open('/data/pictures_annotate_feature/sample_list.p', 'wb'))
    save_annotate_data()
    # merge_data()
    # save_annotate_person_feature()
    # save_vgg_data_feature()
