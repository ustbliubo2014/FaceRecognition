# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: test_recognize_server_v3.py
@time: 2016/11/18 11:50
@contact: ustb_liubo@qq.com
@annotation: test_recognize_server_v3
"""
import sys
import logging
from logging.config import fileConfig
import os
import requests
import urllib
import urllib2
import pdb
import numpy as np
import cv2
import base64
import time
from scipy.misc import imread
import zlib

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    pic_path = sys.argv[1]
    start = time.time()
    face_array = imread(pic_path)
    # face_array : RGB, dtype = int8
    face_array_str = base64.encodestring(zlib.compress(cv2.imencode('.jpg', face_array)[1].tostring()))
    request = {
        "image_id": 'test',
        "image": face_array_str,
        "request_type": 'recognization'
    }
    result = requests.post("http://10.160.164.26:7777/", data=request)
    print result.content[0:20], (time.time() - start)

