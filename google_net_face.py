# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: google_net_face.py
@time: 2016/7/22 10:16
@contact: ustb_liubo@qq.com
@annotation: google_net_face
"""
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='google_net_face.log',
                    filemode='a+')


import os
import sys
from CNN_Model.google_net import train_valid_model
from sklearn.cross_validation import train_test_split
from util.load_data import load_rgb_multi_person_all_data
from DeepId_annotate import load_train_data

class GoogleFace():
    def __init__(self, model_type):
        self.data_folder = '/data/liubo/face/vgg_face_dataset/'
        self.model_folder = '/data/liubo/face/vgg_face_dataset/model/'
        self.small_person_num_str = 'small_data'
        self.all_person_num_str = 'all_data'
        self.person_num = 500
        str_value = {'person_num': self.person_num}
        if model_type == 'new_shape':
            self.pic_shape = (50, 50, 3)
            self.model_file = os.path.join(self.model_folder, 'annotate.%(person_num)d.new_shape.rgb.google_net.model') % str_value
            self.weight_file = os.path.join(self.model_folder,'annotate.%(person_num)d.new_shape.rgb.google_net.weight')%str_value
            self.part_func = None
        else:
            print 'error para'
            sys.exit()
        print self.model_file
        print self.weight_file

    def train_vgg_face(self):
        # 所有数据进行训练
        data, label = load_train_data('/data/pictures_annotate', shape=self.pic_shape)

        data = np.transpose(data, (0, 3, 1, 2))
        X_train, X_test, y_train,  y_test = train_test_split(data, label, test_size=0.1)
        nb_classes = len(set(list(y_train)))
        # nb_classes = 2622
        # pdb.set_trace()

        print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape
        weight_file = self.weight_file
        model_file = self.model_file
        print 'model_file : ', model_file, ' weight_file : ', weight_file
        train_valid_model(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file)



if __name__ == '__main__':
    model_type = 'new_shape'

    google_face = GoogleFace(model_type)
    google_face.train_vgg_face()

