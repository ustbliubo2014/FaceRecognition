# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: test_detect_recognize_server.py
@time: 2016/11/25 12:35
@contact: ustb_liubo@qq.com
@annotation: test_detect_recognize_server
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

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    pic_path = '1478250617.25.jpg'
    start = time.time()
    pic_binary_data = open(pic_path, 'rb').read()
    result = requests.post("http://10.160.164.26:7777/", pic_binary_data)
    # result = requests.post("http://olgpu10.ai.shbt.qihoo.net:8001/test.html", pic_binary_data)
    print result.content
    print (time.time() - start)