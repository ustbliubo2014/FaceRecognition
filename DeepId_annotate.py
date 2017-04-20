# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_annotate.py
@time: 2016/8/2 15:58
@contact: ustb_liubo@qq.com
@annotation: DeepId_annotate
"""
import sys
import logging
from logging.config import fileConfig
import os
import os
import sys
from DeepID.DeepId1.DeepId import train_valid_deepid, extract_feature
import numpy as np
from time import time
from sklearn.cross_validation import train_test_split
import traceback
import pdb
from util.load_data import load_rgb_multi_person_all_data, read_one_rgb_pic, load_two_deep_path

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

def load_train_data(folder, shape=(78, 62, 3)):
    data = []
    label = []
    person_path_dic = load_two_deep_path(folder)
    current_id = 0
    # channel_sum = [0, 0, 0]
    channel_mean = np.array([ 143.69581142,  113.83085749,  100.20530457])
    for person in person_path_dic:
        pic_path_list = person_path_dic.get(person)
        for pic_path in pic_path_list:
            pic_arr = read_one_rgb_pic(pic_path, pic_shape=shape, func_args_dic={})
            pic_arr_mean = pic_arr - channel_mean
            data.append(pic_arr_mean)
            label.append(current_id)
        current_id += 1
    data = np.asarray(data) / 255.0
    label = np.asarray(label)
    return data, label



class DeepId():
    def __init__(self):
        self.data_folder = '/data/liubo/face/vgg_face_dataset/'
        self.model_folder = '/data/liubo/face/vgg_face_dataset/model/'
        self.small_person_num_str = 'small_data'
        self.all_person_num_str = 'all_data'
        self.pic_shape = (128, 128, 3)
        self.deepid_model_file = os.path.join(self.model_folder, 'annotate.all_data.mean.rgb.deepid_relu.deep_filters.model')
        self.deepid_weight_file = os.path.join(self.model_folder,'annotate.all_data.mean.rgb.deepid_relu.deep_filters.weight')
        print self.deepid_model_file
        print self.deepid_weight_file

    def train_vgg_face(self):
        # 所有数据进行训练
        data_folder = self.data_folder
        data, label = load_train_data('/data/pictures_annotate', shape=self.pic_shape)

        data = np.transpose(data, (0, 3, 1, 2))
        X_train, X_test, y_train,  y_test = train_test_split(data, label, test_size=0.1)
        nb_classes = len(set(list(y_train)))

        print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape
        weight_file = self.deepid_weight_file
        model_file = self.deepid_model_file
        print 'model_file : ', model_file, ' weight_file : ', weight_file
        error_valid_sample_file = os.path.join(data_folder,'error_valid_sample.txt')
        error_train_sample_file = os.path.join(data_folder,'error_train_sample.txt')
        train_valid_deepid(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file,
                       error_train_sample_file, error_valid_sample_file)


if __name__ == '__main__':
    deepid = DeepId()
    deepid.train_vgg_face()


