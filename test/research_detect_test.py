# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: research_detect_test.py
@time: 2016/11/16 14:53
@contact: ustb_liubo@qq.com
@annotation: research_detect_test
"""
import sys
import logging
from logging.config import fileConfig
import os
import requests
import cv2

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    # curl --data-binary @1478257275.37.jpg "olgpu10.ai.shbt.qihoo.net:8001/test.html"
    pic_path = 'face_cv2.jpg'

    pic_binary_data = open(pic_path, 'rb').read()
    request = requests.post("http://olgpu10.ai.shbt.qihoo.net:8001/test.html", pic_binary_data)
    print request.status_code, request.content
    tmp = request.content.split('\n')
    frame = map(float, tmp[1].split(','))
    img = cv2.imread(pic_path)
    new_img = img[int(frame[0]): int(frame[3]), int(frame[1]):int(frame[2]), :]
    cv2.imwrite('new_face.jpg', new_img)

