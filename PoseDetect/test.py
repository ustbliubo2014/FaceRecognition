# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: test.py
@time: 2017/1/23 11:30
@contact: ustb_liubo@qq.com
@annotation: test
"""
import sys
import logging
from logging.config import fileConfig
import os
import detect
import cv2
import pdb
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    # 人脸姿势计算和人脸转换
    # 流程:1.dlib进行人脸检测; 2.计算pose; 3.人脸转正
    img = cv2.imread('1479724298.1.jpg')
    print img.shape
    # all_face : (x1, y1, x2, y2)
    faces = detect.faces(img)
    landmarks = detect.landmarks(img, faces[0])
    print detect.face_pose(img, faces[0], np.asarray([(element.x, element.y) for element in landmarks.parts()]))
    # [(element.x, element.y) for element in landmarks.parts()]
