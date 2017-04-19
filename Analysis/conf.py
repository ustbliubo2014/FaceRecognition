# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: conf.py
@time: 2016/11/16 11:12
@contact: ustb_liubo@qq.com
@annotation: conf
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


class SkyEyeConf():
    def __init__(self):
        self.feature_url = "http://10.160.164.26:7777/"
        self.feature_dim = 512  # 以后调模型时不在修改最后一个卷积层的维度
        # 每次更换模型的时候需要修改这两个参数
        self.same_pic_threshold = 0.8
        self.upper_threshold = 0.65
        self.lower_threshold = 0.45
        self.verification_model_file = '/home/liubo/FaceRecognition/data/model/skyeye_verification_light_cnn_model'
        self.all_feature_label_file = '/home/liubo/FaceRecognition/data/feature/skyeye_all_feature_label.p'
        self.log_dir = 'skyeye_log/'
        self.model_label = 'skyeye'


class ResearchConf():
    def __init__(self):
        self.feature_url = "http://olgpu10.ai.shbt.qihoo.net:8001/test.html"
        self.feature_dim = 256  # 以后调模型时不在修改最后一个卷积层的维度
        # 每次更换模型的时候需要修改这两个参数
        self.same_pic_threshold = 0.85
        self.upper_threshold = 0.8
        self.lower_threshold = 0.65
        self.verification_model_file = '/home/liubo/FaceRecognition/data/model/research_verification_model'
        self.all_feature_label_file = '/home/liubo/FaceRecognition/data/feature/research_all_feature_label.p'
        self.log_dir = 'research_log/'
        self.model_label = 'institute'


if __name__ == '__main__':
    pass
