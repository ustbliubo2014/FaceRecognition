# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: conf.py
@time: 2016/7/27 15:07
@contact: ustb_liubo@qq.com
@annotation: conf
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


data_folder = '/data/liubo/face/vgg_face_dataset/'
model_folder = '/data/liubo/face/vgg_face_dataset/model/'
code_folder = '/home/liubo/kaggle_sadsb/restructure/'
raw_data_folder = '/data/liubo/face/self'

if __name__ == '__main__':
    pass
