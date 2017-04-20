# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: align_interface.py
@time: 2016/11/2 11:37
@contact: ustb_liubo@qq.com
@annotation: align_interface
"""
import sys
import logging
from logging.config import fileConfig
import os
from align_face import AlignDlib
import cv2
from scipy.misc import imsave
import numpy as np
from time import time
import pdb
import shutil
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


opencv_model = '/data/liubo/face/annotate_face_model/cascade.xml'
dlibFacePredictor = '/data/liubo/face/annotate_face_model/shape_predictor_68_face_landmarks.dat'
landmarks = 'innerEyesAndBottomLip'
ts = 0.1
size = 96
align = AlignDlib(dlibFacePredictor)
landmarkIndices = AlignDlib.INNER_EYES_AND_BOTTOM_LIP

def getBGR(path):
    """
    Load the image from disk in BGR format.

    :return: BGR image. Shape: (height, width, 3)
    :rtype: numpy.ndarray
    """
    try:
        bgr = cv2.imread(path)
    except:
        bgr = None
    return bgr


def getRGB(path):
    """
    Load the image from disk in RGB format.

    :return: RGB image. Shape: (height, width, 3)
    :rtype: numpy.ndarray
    """
    bgr = getBGR(path)
    # width = min(bgr.shape[0], 192)
    # height = min(bgr.shape[1], 192)
    # bgr = cv2.resize(bgr, (width, height))
    if bgr is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = None
    return rgb


def align_face(path):
    return align_face_rgb_array(getRGB(path))


def align_face_rgb_array(rgb_array, bb=None):
    # 对齐rgb格式的人脸矩阵, 返回BRG格式的数据
    try:
        outRgb = align.align(size, rgb_array, bb=bb, ts=None,
                             landmarks=None, landmarkIndices=landmarkIndices,
                             opencv_det=True, opencv_model=opencv_model,
                             only_crop=False)
        face = cv2.cvtColor(outRgb, cv2.COLOR_BGR2RGB)
        return face
    except:
        return None


def detect_face_rgb_array(rgb_array, bb=None):
    # 对齐rgb格式的人脸矩阵, 返回BRG格式的数据
    try:
        outRgb = align.detect(size, rgb_array, bb=bb, ts=None,
                             landmarks=None, landmarkIndices=landmarkIndices,
                             opencv_det=False, opencv_model=opencv_model,
                             only_crop=False)
        face = cv2.cvtColor(outRgb, cv2.COLOR_BGR2RGB)
        return face
    except:
        return None


if __name__ == '__main__':

    # path = sys.argv[1]
    # img = cv2.imread(path)
    # img = cv2.resize(img, (214, 120))
    # print img.shape
    # start = time()
    # rgb_face = align_face_rgb_array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # end = time()
    # print rgb_face.shape, (end - start)
    # cv2.imwrite('face_cv2.png', rgb_face)

    size = 96
    folder = '/data/liubo/face/baihe/person'
    align_folder = '/data/liubo/face/baihe/person_dlib_face'
    if not os.path.exists(align_folder):
        os.makedirs(align_folder)
    person_list = os.listdir(folder)
    for person in person_list:
        try:
            src_person_path = os.path.join(folder, person)
            dst_person_path = os.path.join(align_folder, person)
            print 'person name :', person
            if not os.path.exists(dst_person_path):
                os.makedirs(dst_person_path)
            else:
                continue
            pic_list = os.listdir(src_person_path)
            for pic in pic_list:
                try:
                    src_pic_path = os.path.join(src_person_path, pic)
                    dst_pic_path = os.path.join(dst_person_path, pic)
                    if os.path.exists(dst_pic_path):
                        continue
                    face = align_face(src_pic_path)
                    if face == None:
                        # shutil.copy(src_pic_path, dst_pic_path)
                        continue
                    else:
                        cv2.imwrite(dst_pic_path, face)
                except:
                    traceback.print_exc()
                    continue
        except:
            traceback.print_exc()
            continue

