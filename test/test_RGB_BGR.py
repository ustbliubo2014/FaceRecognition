# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: test_RGB_BGR.py
@time: 2016/11/18 11:15
@contact: ustb_liubo@qq.com
@annotation: test_RGB_BGR
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


from scipy.misc import imread
import cv2
import pdb

# cv2.imread 读入的数据是BGR通道, scipy.misc.imread读入的是RGB通道
# 转换方法: arr4=arr2[:, :, ::-1]
# caffe 读入数据是BGR格式
pic_path = '1478250742.82.jpg'
arr1 = cv2.imread(pic_path)
arr2 = imread(pic_path)
arr3 = cv2.cvtColor(arr2, cv2.COLOR_RGB2BGR)
arr4 = cv2.cvtColor(arr1, cv2.COLOR_BGR2RGB)
pdb.set_trace()
