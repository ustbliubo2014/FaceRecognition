# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: vgg_annotate.py
@time: 2016/8/8 18:17
@contact: ustb_liubo@qq.com
@annotation: vgg_annotate
"""
import sys
import logging
from logging.config import fileConfig
import os
from CNN_Model.deep_face import train_valid_model
import numpy as np
from util.load_data import split_data, load_two_deep_path
from lfw_vgg_valid import read_one_rgb_pic
import pdb
import msgpack

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def load_train_data(folder, shape=(224, 224, 3)):
    data = []
    label = []
    person_path_dic = load_two_deep_path(folder)
    current_id = 0
    # 使用vgg的读取方法
    for person in person_path_dic:
        pic_path_list = person_path_dic.get(person)
        for pic_path in pic_path_list:
            pic_arr = read_one_rgb_pic(pic_path, pic_shape=shape)
            data.append(pic_arr)
            label.append(current_id)
        current_id += 1
    data = np.asarray(data)
    label = np.asarray(label)
    return data, label


class DeepFace():
    def __init__(self):
        self.data_folder = '/data/liubo/face/vgg_face_dataset/'
        self.model_folder = '/data/liubo/face/vgg_face_dataset/model/'
        self.small_person_num_str = 'small_data'
        self.all_person_num_str = 'all_data'
        self.person_num = 554
        self.pic_shape = (224, 224, 3)
        self.model_file = os.path.join(self.model_folder, 'annotate_deep_face.model')
        self.weight_file = os.path.join(self.model_folder,'annotate_deep_face.weight')
        self.part_func = None
        print self.model_file
        print self.weight_file

    def train_vgg_face(self):
        # 所有数据进行训练

        train_path_list, valid_path_list = msgpack.load(open('/data/annotate_list.p'))
        nb_classes = max(set(map(lambda x:x[1], valid_path_list)) | set(map(lambda x:x[1], train_path_list))) + 1
        input_shape = (self.pic_shape[2], self.pic_shape[0], self.pic_shape[1])
        weight_file = self.weight_file
        model_file = self.model_file
        print 'nb_classes :', nb_classes, 'model_file : ', model_file, ' weight_file : ', weight_file, 'input_shape :', input_shape
        train_valid_model(train_path_list, valid_path_list, self.pic_shape , nb_classes, model_file, weight_file)


if __name__ == '__main__':

    deep_face = DeepFace()
    deep_face.train_vgg_face()

