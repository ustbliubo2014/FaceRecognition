#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: util.py
@time: 2016/6/2 11:43
@contact: ustb_liubo@qq.com
@annotation: util
"""

import math
import time
import cv2
from scipy.misc import imread, imresize
import numpy as np
import traceback
import urllib2
import urllib
import pdb
import os
import shutil
import sys


avg = np.array([129.1863, 104.7624, 93.5940])
blur_threshold = 100


def cal_distance(vec_tup):
    vec1, vec2 = vec_tup
    vec1 = vec1[0]
    vec2 = vec2[0]
    return math.sqrt(sum((a - b)**2 for a, b in zip(vec1, vec2)))


def is_blur(image):
    try:
        cut = 3
        width = image.shape[0] * 1.0 / cut
        height = image.shape[1] * 1.0 / cut
        # 将图片分成9个区域，分别计算清晰度，选最小的一个
        var_list = []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for index_w in range(cut):
            for index_h in range(cut):
                var = cv2.Laplacian(image[index_w*width:(index_w+1)*width, index_h*height:(index_h+1)*height], cv2.CV_64F).var()
                var_list.append(var)
        var_list.sort()
        check_var = np.mean(var_list[:3])
        if check_var > blur_threshold:
            return False, check_var
        else:
            return True, check_var
    except:
        traceback.print_exc()
        return None


def read_one_rgb_pic(pic_path, pic_shape):
    # 读入的是RGB格式的array
    img = imresize(imread(pic_path), pic_shape)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    # 转换成BGR格式
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def normalize_rgb_array(img):
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def image_request(request, url):
    try:
        requestPOST = urllib2.Request(
            data=urllib.urlencode(request),
            url=url
        )
        requestPOST.get_method = lambda: "POST"
    except:
        return None
    try:
        result = urllib2.urlopen(requestPOST).read()
        return result
    except urllib2.HTTPError, e:
        print e.code
    except urllib2.URLError, e:
        print str(e)
    return None


def get_time_slot(time_stamp):
    # 十分钟一个文件夹
    # timeStamp = 1478774205.86
    try:
        time_array = time.localtime(float(time_stamp))
        otherStyleTime = time.strftime("%Y-%m-%d-%H-%M", time_array)
        tmp = otherStyleTime.split('-')
        tmp[-1] = str(int(tmp[-1]) / 10)+'0'
        otherStyleTime = '-'.join(tmp)
        return otherStyleTime
    except:
        return None


def get_current_day():
    time_array = time.localtime(time.time())
    current_day = time.strftime("%Y-%m-%d", time_array)
    return current_day


def get_current_time():
    time_array = time.localtime(time.time())
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    return current_time


if __name__ == '__main__':
    # print is_blur(cv2.imread('C:\Users\liubo\Desktop\picture\wangzhanyi/1478771705.15.jpg'))
    # print is_blur(cv2.imread('C:\Users\liubo\Desktop/test.png'))
    start = time.time()

    sign, smallest_var = is_blur(cv2.resize(cv2.imread('C:\Users\liubo\Desktop/zhangshunlong_test.png'), (96, 96)))
    end = time.time()
    print (end - start), smallest_var
    # print cv2.imwrite('C:\Users\liubo\Desktop/test_96.jpg', cv2.resize(cv2.imread('C:\Users\liubo\Desktop/test.png'), (96, 96)))