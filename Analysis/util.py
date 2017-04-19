#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: util.py
@time: 2016/11/16 16:13
@contact: ustb_liubo@qq.com
@annotation: util
"""

import time
import cv2
import numpy as np
import traceback
import pdb
import os
import shutil
import base64


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

    print get_current_day()
    print (get_current_time())
    start = time.time()
    for index in range(1000):
        get_current_time()
    print time.time() - start
    # base64.encodestring(cv2.imencode('.jpg', face)[1].tostring())

    # 解析数据库里的人脸图片
    image = open('pic.txt').read().split()[1]
    image = base64.decodestring(image)
    im = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1)
    print im.shape
    cv2.imwrite('face1.jpg', im)