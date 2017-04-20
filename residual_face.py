# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: residual_face.py
@time: 2016/7/18 9:50
@contact: ustb_liubo@qq.com
@annotation: residual_face
"""


import os
import sys
from CNN_Model.residual import train_valid_model
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
    def __init__(self, model_type, person_num):
        self.data_folder = '/data/liubo/face/vgg_face_dataset/'
        self.model_folder = '/data/liubo/face/vgg_face_dataset/model/'
        self.small_person_num_str = 'small_data'
        self.all_person_num_str = 'all_data'
        self.person_num = person_num
        if model_type == 'new_shape':
            self.pic_shape = (78, 62, 3)
            self.person_num = person_num
            str_value = {'person_num':self.person_num}
            if self.person_num <= 300:
                self.pic_num_threshold = None
                self.model_file = os.path.join(self.model_folder, 'vgg_face.%(person_num)d.new_shape.rgb.residual.model')%str_value
                self.weight_file = os.path.join(self.model_folder,'vgg_face.%(person_num)d.new_shape.rgb.residual.weight')%str_value
                self.part_func = None
            elif self.person_num == 2600:
                self.pic_num_threshold = 30
                self.model_file = os.path.join(self.model_folder, 'vgg_face.all_data.new_shape.rgb.residual.model')
                self.weight_file = os.path.join(self.model_folder,'vgg_face.all_data.new_shape.rgb.residual.weight')
                self.part_func = None
        else:
            print 'error para'
            sys.exit()
        print self.model_file
        print self.weight_file

    def train_vgg_face(self):
        # 所有数据进行训练
        data, label = load_train_data('/data/liubo/face/vgg_face_dataset/all_data/pictures_box', shape=self.pic_shape,
                        pic_num_threshold=self.pic_num_threshold, part_func=self.part_func, person_num=self.person_num)

        X_train, X_test, y_train,  y_test = train_test_split(data, label, test_size=0.05)
        nb_classes = len(set(list(y_train)))
        # nb_classes = 2622
        # pdb.set_trace()

        print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape
        weight_file = self.weight_file
        model_file = self.model_file
        print 'model_file : ', model_file, ' weight_file : ', weight_file
        train_valid_model(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file)


if __name__ == '__main__':
    model_type = sys.argv[1]
    person_num = int(sys.argv[2])

    residual_face = ResidualFace(model_type, person_num)
    residual_face.train_vgg_face()

