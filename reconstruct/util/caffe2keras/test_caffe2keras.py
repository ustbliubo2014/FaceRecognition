# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: test_caffe2keras.py
@time: 2016/8/8 12:24
@contact: ustb_liubo@qq.com
@annotation: test_caffe2keras
"""
import sys
import logging
from logging.config import fileConfig
import os
import unittest
import numpy as np
import deepface
from caffe2keras import Model2Keras

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

prototxt = '/home/liubo-it/VGGFaceModel-master/VGG_FACE_deploy.prototxt'
model_file = '/home/liubo-it/VGGFaceModel-master/VGG_FACE.caffemodel'

model = deepface.deep_face(input_shape=(3, 224, 224), nb_classes=2622)


class TestModel2Keras(unittest.TestCase):
    m2k = Model2Keras(model, prototxt, model_file)

    def test_load_caffe_params(self):
        self.m2k.load_caffe_params()


if __name__ == '__main__':
    print('Test caffe model conversion')
    unittest.main()


