# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: dlib_face_detect.py
@time: 2016/9/23 13:29
@contact: ustb_liubo@qq.com
@annotation: dlib_face_detect
"""
import sys
import logging
from logging.config import fileConfig
import os
from CriticalPointDetection import CriticalPointDetection
import cv2

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    criticalPointDetection = CriticalPointDetection()

    # 用dlib的人脸检测来处理lfw的数据
    lfw_folder = '/data/liubo/face/lfw/'
    lfw_dlib_face_folder = '/data/liubo/face/lfw_dlib_face'
    if not os.path.exists(lfw_dlib_face_folder):
        os.makedirs(lfw_dlib_face_folder)
    person_list = os.listdir(lfw_folder)
    for person in person_list:
        print person
        dlib_person_folder = os.path.join(lfw_dlib_face_folder, person)
        if not os.path.exists(dlib_person_folder):
            os.makedirs(dlib_person_folder)
        lfw_person_folder = os.path.join(lfw_folder, person)
        pic_list = os.listdir(lfw_person_folder)
        for pic in pic_list:
            lfw_pic_path = os.path.join(lfw_person_folder, pic)
            lfw_dlib_pic_path = os.path.join(dlib_person_folder, pic)
            img = cv2.imread(lfw_pic_path)
            face = criticalPointDetection.detect_face(img)
            if face != None:
                cv2.imwrite(lfw_dlib_pic_path, face)
