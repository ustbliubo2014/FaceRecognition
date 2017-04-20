# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: critical_point_detect.py
@time: 2016/7/4 22:17
"""

import logging
import os

if not os.path.exists('log'):
    os.mkdir('log')

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/critical_point_detect.log',
                    filemode='w')


import dlib
import numpy as np
from scipy.misc import imread, imsave, imresize
import math
from skimage.transform import rotate
import pdb
from PIL import Image


PREDICTOR_PATH = '/data/liubo/face/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
FACE_POINTS = list(range(17, 68))
MOUTH_POINT = 51
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINT = 42
LEFT_EYE_POINTS = list(range(42, 48))
LEFT_EYE_POINT = 39
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))


def cal_angel(landmarks):
    # 计算图片偏移的角度
    right_eye = np.asarray([landmarks[42][0, 0], landmarks[42][0, 1]])
    left_eye = np.asarray([landmarks[36][0, 0], landmarks[36][0, 1]])
    angle = np.arctan((right_eye[1]-left_eye[1])*1.0/(right_eye[0]-left_eye[0])) * 180 / np.pi
    return angle


def get_landmarks(im):
    return np.matrix([[p.x, p.y] for p in predictor(im, dlib.rectangle(0, 0, im.shape[0], im.shape[1])).parts()])



def get_left_eye(im, new_pic_shape):
    if type(im) == np.ndarray and len(im.shape) == 3:
        # rgb图像
        gray_im = np.array(Image.fromarray(np.uint8(im*255)).convert('L'))
        landmarks = get_landmarks(gray_im)
        angle = cal_angel(landmarks)
        im_rotate = rotate(im, angle)
        left_eye = im_rotate[:, :np.array(landmarks[RIGHT_EYE_POINT])[0][1]]
        return imresize(left_eye, new_pic_shape)
    landmarks = get_landmarks(im)
    angle = cal_angel(landmarks)
    im_rotate = rotate(im, angle)
    left_eye = im_rotate[:, :np.array(landmarks[RIGHT_EYE_POINT])[0][1]]
    return imresize(left_eye, new_pic_shape)


def get_right_eye(im, new_pic_shape):
    if type(im) == np.ndarray and len(im.shape) == 3:
        # rgb图像
        gray_im = np.array(Image.fromarray(np.uint8(im*255)).convert('L'))
        landmarks = get_landmarks(gray_im)
        angle = cal_angel(landmarks)
        im_rotate = rotate(im, angle)
        right_eye = im_rotate[:, np.array(landmarks[LEFT_EYE_POINT])[0][1]:]
        return imresize(right_eye, new_pic_shape)
    landmarks = get_landmarks(im)
    angle = cal_angel(landmarks)
    im_rotate = rotate(im, angle)
    right_eye = im_rotate[:, np.array(landmarks[LEFT_EYE_POINT])[0][1]:]
    return imresize(right_eye, new_pic_shape)


def get_nose(im, new_pic_shape):
    if type(im) == np.ndarray and len(im.shape) == 3:
        # rgb图像
        gray_im = np.array(Image.fromarray(np.uint8(im*255)).convert('L'))
        landmarks = get_landmarks(gray_im)
        angle = cal_angel(landmarks)
        im_rotate = rotate(im, angle)
        nose_im = im_rotate[:-np.array(landmarks[MOUTH_POINT])[0][0], :]
        return imresize(nose_im, new_pic_shape)
    landmarks = get_landmarks(im)
    angle = cal_angel(landmarks)
    im_rotate = rotate(im, angle)
    nose_im = im_rotate[:-np.array(landmarks[MOUTH_POINT])[0][0], :]
    return imresize(nose_im, new_pic_shape)



if __name__ == '__main__':
    pic_path = '/data/liubo/face/rotate_test/1466561833.13.png'
    im = np.array(Image.open(pic_path).convert('L'))
    print im.shape
    landmarks = get_landmarks(im)
    angle = cal_angel(landmarks)
    rotate_im = rotate(im, angle)
    rotate_im = np.array(Image.fromarray(np.uint8(rotate_im*255)))
    print rotate_im.shape
    new_landmarks = get_landmarks(rotate_im)
    imsave('/data/liubo/face/rotate_test/rotate_im.png', rotate_im)
    print angle
    # get_nose(rotate_im, new_landmarks)
    imsave('left_eye.png', get_left_eye(im))
    imsave('right_eye.png', get_right_eye(im))
    imsave('nose.png', get_nose(im))

