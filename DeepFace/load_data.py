# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: load_data.py
@time: 2016/8/8 19:02
@contact: ustb_liubo@qq.com
@annotation: load_data
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
from scipy.misc import imread, imsave, imresize
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

avg = np.array([129.1863, 104.7624, 93.5940])
pic_shape = (224, 224, 3)

def read_one_rgb_pic(pic_path, pic_shape=pic_shape):
    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def load_data_from_list(sample_list, pic_shape=pic_shape):
    all_data = []
    all_label = []
    for pic_path, pic_label in sample_list:
        img = read_one_rgb_pic(pic_path, pic_shape)[0]
        all_data.append(img)
        all_label.append(pic_label)
    all_label = np.asarray(all_label)
    all_data = np.asarray(all_data)
    return all_data, all_label


if __name__ == '__main__':
    pass
