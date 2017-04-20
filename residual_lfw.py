# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: residual_lfw.py
@time: 2016/7/18 12:19
@contact: ustb_liubo@qq.com
@annotation: residual_lfw
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='residual_lfw.log',
                    filemode='a+')

import os
import sys
from CNN_Model.residual import train_valid_model, extract_feature
import numpy as np
from time import time
from sklearn.cross_validation import train_test_split
from PIL import Image
from scipy.misc import imread, imsave, imresize
from CriticalPointDetection.split_pic import get_landmarks, get_nose, get_left_eye, get_right_eye, cal_angel
import traceback
from skimage.transform import rotate
import pdb
from util.load_data import load_rgb_multi_person_all_data


def load_train_data(folder, shape=(156, 124, 3), need_pic_list=False, pic_num_threshold=1000, part_func=None,
                    person_num=None):
    label_int = True
    filter_list = []
    func_args_dic = {}
    data, label, all_pic_list = load_rgb_multi_person_all_data(
                        folder, shape, label_int, person_num, pic_num_threshold, filter_list, func_args_dic)
    if need_pic_list:
        return data, label, all_pic_list
    else:
        return data, label


class ResidualFace():
    def __init__(self):
        self.data_folder = '/data/liubo/face/vgg_face_dataset/'
        self.model_folder = '/data/liubo/face/vgg_face_dataset/model/'
        self.pic_num_threshold = 300
        self.small_person_num_str = 'small_data'
        self.all_person_num_str = 'all_data'
        self.pic_shape = (78, 62, 3)
        self.person_num = 10000
        self.model_file = os.path.join(self.model_folder, 'lfw.rgb.residual.model')
        self.weight_file = os.path.join(self.model_folder,'lfw.rgb.residual.weight')
        print self.model_file
        print self.weight_file
        self.part_func = None


    def train_vgg_face(self):
        # 所有数据进行训练
        data, label = load_train_data('/data/liubo/face/lfw_face', shape=self.pic_shape,
                        pic_num_threshold=self.pic_num_threshold, part_func=self.part_func, person_num=self.person_num)

        X_train, X_test, y_train,  y_test = train_test_split(data, label, test_size=0.1)
        nb_classes = np.max([np.max(y_train), np.max(y_test)]) + 1
        # nb_classes = 2622
        # pdb.set_trace()

        print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape
        weight_file = self.weight_file
        model_file = self.model_file
        print 'model_file : ', model_file, ' weight_file : ', weight_file
        train_valid_model(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file)

    def extract_feature(self):
        # 所有数据进行训练
        data, label = load_train_data('/data/liubo/face/lfw_face', shape=self.pic_shape,
                        pic_num_threshold=self.pic_num_threshold, part_func=self.part_func, person_num=self.person_num)

        X_train, X_test, y_train,  y_test = train_test_split(data, label, test_size=0.1)
        nb_classes = np.max([np.max(y_train), np.max(y_test)]) + 1
        # nb_classes = 2622
        # pdb.set_trace()

        print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape
        weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.new_shape.rgb.residual.weight'
        model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.new_shape.rgb.residual.model'
        print 'model_file : ', model_file, ' weight_file : ', weight_file
        model, get_Conv_FeatureMap = extract_feature(model_file, weight_file)
        data_feature = np.zeros(shape=(data.shape[0], 1024))



if __name__ == '__main__':
    residual_face = ResidualFace()
    residual_face.train_vgg_face()
    # residual_face.extract_feature()

