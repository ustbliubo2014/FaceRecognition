# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: split_pic.py
@time: 2016/6/29 19:22
@contact: ustb_liubo@qq.com
@annotation: split_pic
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import dlib
import numpy as np
from scipy.misc import imread, imsave, imresize
import math
from skimage.transform import rotate
import pdb
from PIL import Image


PREDICTOR_PATH = '/home/liubo-it/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
FACE_POINTS = list(range(17, 68))
MOUTH_POINT = 51
MOUTH_IMG_SIZE = (88, 128)
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINT = 42
EYE_IMG_SIZE = (128, 88)
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



def get_left_eye(im_rotate, landmarks):
    '''
        以左眼为中心, 划分区域
        :param im_rotate:原始图片(选择之后,方便切分)
        :param landmarks:所有关键点
    :return:
    '''
    left_eye = im_rotate[:, :np.array(landmarks[RIGHT_EYE_POINT])[0][1]]
    return imresize(left_eye, im_rotate.shape)


def get_right_eye(im_rotate, landmarks):
    '''
        以右眼为中心, 划分区域
        :param im_rotate:原始图片(选择之后,方便切分)
        :param landmarks:所有关键点
    :return:
    '''
    right_eye = im_rotate[:, np.array(landmarks[LEFT_EYE_POINT])[0][1]:]
    return imresize(right_eye, im_rotate.shape)


def get_nose(im_rotate, landmarks):
    '''
        以鼻子为中心, 划分区域
        :param im_rotate:原始图片(选择之后,方便切分)
        :param landmarks:所有关键点
    :return:
    '''

    nose_im = im_rotate[:-np.array(landmarks[MOUTH_POINT])[0][0], :]
    return imresize(nose_im, im_rotate.shape)





if __name__ == '__main__':
    pic_path = '/data/liubo/face/vgg_face_dataset/all_data/pictures_box/Karla_Souza/00000793.png'
    im = np.array(Image.open(pic_path))
    print im.shape
    landmarks = get_landmarks(im)
    angle = cal_angel(landmarks)
    rotate_im = rotate(im, angle)
    print rotate_im.shape
    new_landmarks = get_landmarks(rotate_im)
    print angle
    # get_nose(rotate_im, new_landmarks)
    imsave('left_eye.png', get_left_eye(rotate_im, new_landmarks))
    imsave('right_eye.png', get_right_eye(rotate_im, new_landmarks))
    imsave('nose.png', get_nose(rotate_im, new_landmarks))

