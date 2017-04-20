# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/11/4 11:37
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import logging
from logging.config import fileConfig
import os
import cv2
from collections import deque
import numpy as np
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

blur_threshold = 20

def is_blur(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(image, cv2.CV_64F).var()
    print 'blur_var :', var
    if var > blur_threshold:
        return False
    else:
        return True

# image = cv2.imread('10.74.104.113_01_20161104123346767.jpg')
# is_blur(image)
# is_blur(image[350:650, 350:650,:])
# image = cv2.imread('abc_1478226833.94.jpg.jpg')
# is_blur(image)

def size_stat():

    folder = '/tmp/face_recog_tmp'
    pic_list = map(lambda x:os.path.join(folder, x), os.listdir(folder))
    size_stat_dic = {}
    for pic in pic_list:
        try:
            shape = cv2.imread(pic).shape
            size_stat_dic[shape] = size_stat_dic.get(shape, 0) + 1
        except:
            continue
    print size_stat_dic


def deque_test():
    _d = deque(maxlen=5)
    _d.append(1)
    _d.append(2)
    _d.append(3)
    _d.append(np.asarray(range(4096)))
    _d.append(np.asarray(range(4096, 8192)))
    _d.append(range(8192, 12288))
    pdb.set_trace()

if __name__ == '__main__':
    # size_stat()
    deque_test()
