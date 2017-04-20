# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: glass_classify.py
@time: 2016/8/24 14:45
@contact: ustb_liubo@qq.com
@annotation: glass_classify
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
from skimage.feature import hog
from scipy.misc import imread, imresize
import shutil
import msgpack_numpy
from time import time
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import cPickle
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")


has_glass = 0
no_glass = 1
model_file = '/data/liubo/face/glass_data/model.p'
pic_shape = (224, 224)


def normHOG(img_arr):
    # img_arr = imresize(imread(images_file, mode='L'), size=pic_shape)
    width, height = img_arr.shape
    f = hog(img_arr, pixels_per_cell=(height//4, width//4))
    return f.reshape(-1)


def load_data():
    glass_folder = '/data/liubo/face/glass_data/glass_face'
    no_glass_folder = '/data/liubo/face/glass_data/no_glass_data'
    glass_pic_list = os.listdir(glass_folder)
    no_glass_pic_list = os.listdir(no_glass_folder)
    all_data = []
    all_label = []
    for pic in glass_pic_list:
        img_arr = imresize(imread(os.path.join(glass_folder, pic), mode='L'), size=pic_shape)
        hog_feature = normHOG(img_arr)
        all_data.append(hog_feature)
        all_label.append(has_glass)
    for pic in no_glass_pic_list:
        img_arr = imresize(imread(os.path.join(no_glass_folder, pic), mode='L'), size=pic_shape)
        hog_feature = normHOG(img_arr)
        all_data.append(hog_feature)
        all_label.append(no_glass)
    all_data = np.asarray(all_data)
    all_label = np.asarray(all_label)
    train_data, valid_data, train_label, valid_label = train_test_split(all_data, all_label, test_size=0.9)
    return train_data, valid_data, train_label, valid_label


def train_valid():
    clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=100)
    # clf = SVC(kernel='linear', probability=True)
    train_data, valid_data, train_label, valid_label = load_data()
    print train_data.shape, valid_data.shape
    clf.fit(train_data, train_label)
    valid_acc = accuracy_score(valid_label, clf.predict(valid_data))
    train_acc = accuracy_score(train_label, clf.predict(train_data))
    cPickle.dump(clf, open(model_file, 'wb'))
    print train_acc, valid_acc


if __name__ == '__main__':
    # train_valid()
    #
    clf = cPickle.load(open(model_file, 'rb'))
    # print clf.predict_proba(normHOG(imresize(imread('liubo-it1468381751.27.png_face_0.jpg', mode='L'), size=pic_shape)))
    folder = 'glass_data/glass_face'
    person_list = os.listdir(folder)
    pic_list = os.listdir(folder)
    for pic in pic_list:
        pic_path = os.path.join(folder, pic)
        feature = normHOG(imresize(imread(pic_path, mode='L'), size=pic_shape))
        feature = np.reshape(feature, (1, len(feature)))
        print pic, clf.predict(feature)
