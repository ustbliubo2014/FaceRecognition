# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: show_all_images.py
@time: 2016/11/18 18:08
@contact: ustb_liubo@qq.com
@annotation: show_all_images : 将从数据库读出的图片解码
"""
import sys
import logging
from logging.config import fileConfig
import os
import cv2
import base64
import numpy as np
import pdb
import time
from recog_util import is_blur
reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def test1():
    txt_file = '/tmp/all_images.txt'
    folder = '/tmp/annotate/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    for line in open(txt_file):
        tmp = line.rstrip().split('\t')
        img, name, feature, is_moved = tmp
        if img == 'img':
            continue
        person_folder = os.path.join(folder, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        image = base64.decodestring(img)
        im = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1)
        pic_name = os.path.join(person_folder, str(time.time()) + '.jpg')
        cv2.imwrite(pic_name, im)
        time.sleep(0.1)


def filter(folder='/tmp/annotate/'):

    person_list = os.listdir(folder)
    for person in person_list:
        print person
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            img = cv2.imread(pic_path)
            if img.shape[0] < 120 or img.shape[1] < 120 or is_blur(cv2.resize(img, (96, 96)))[0]:
                os.remove(pic_path)

if __name__ == '__main__':
    # test1()
    filter('e:/research_face')
