#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: CriticalPointDetection.py
@time: 2016/5/31 18:37
@contact: ustb_liubo@qq.com
@annotation: test
"""

import numpy as np
import dlib
import cv2
import pdb


class CriticalPointDetection(object):
    def __init__(self):
        self.PREDICTOR_PATH = '/data/liubo/face/model/shape_predictor_68_face_landmarks.dat'
        # self.PREDICTOR_PATH = 'D:/data/face/DlibProject-master/shape_predictor_68_face_landmarks.dat'
        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11

        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

        print 'finish load model'

    def get_landmarks(self, im):
        # im 就是检测到的人脸
        # det = self.detector(im)
        # if det:
        #     return np.matrix([[p.x, p.y] for p in self.predictor(im, det[0]).parts()])

        return np.matrix([[p.x, p.y] for p in self.predictor(im, dlib.rectangle(0, 0, im.shape[0], im.shape[1])).parts()])

    def detect_face(self, img):
        dets = self.detector(img)
        if dets:
            if len(dets) > 1:
                return None
            det = dets[0]
            face = img[det.top():det.bottom(), det.left():det.right(), :]
            return face
        else:
            return None

    def annotate_landmarks(self, im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im

if __name__ == '__main__':
    criticalPointDetection = CriticalPointDetection()
    # im = cv2.imread('1474598545.89_camera.jpg')
    im = cv2.imread('D:\data/face/test_face/wrong/lining3.png_face_0.jpg_face_0.jpg')
    # im = cv2.imread('data/liubo-it1468381755.19.png_face_0.jpg')
    im = cv2.resize(im, (128, 128))
    criticalPoint = criticalPointDetection.get_landmarks(im)
    # print criticalPoint
    annotate_im = criticalPointDetection.annotate_landmarks(im, criticalPoint)
    cv2.imwrite('data/annotate_im_raw_Face4.jpg', annotate_im)
