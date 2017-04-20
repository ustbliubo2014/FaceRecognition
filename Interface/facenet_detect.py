# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: facenet_detect.py
@time: 2017/2/9 12:18
@contact: ustb_liubo@qq.com
@annotation: facenet_detect
"""

import sys
sys.path.insert(0, '/home/liubo-it/FaceRecognization/')
import logging
from logging.config import fileConfig
import os
import cv2
from facenet.src.align import detect_face
import tensorflow as tf
from scipy import misc
from facenet.src import facenet
import pdb
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

minsize = 60  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
image_size = 160
detect_model_dir = '/home/liubo-it/FaceRecognization/facenet/data'
recognize_model_dir = '/home/liubo-it/FaceRecognization/facenet/models/casia_facenet/20170208-100636/valid'


# 由于检测模型和识别模型都比较大,所以分开测试(实际使用时, 放在不同的gpu上)
detect_graph = tf.Graph()
with detect_graph.as_default() as detect_graph:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    detect_session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with detect_session.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(detect_session, detect_model_dir)



def align_face(pic_path):
    if os.path.exists(pic_path):
        try:
            img = misc.imread(pic_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(pic_path, e)
            print(errorMessage)
        if img.ndim < 2:
            print('Unable to align "%s"' % pic_path)
            return
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det = det[index, :]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            misc.imsave(pic_path, scaled)



if __name__ == '__main__':

    folder = '/tmp/test_pic'
    pic_list = map(lambda x: os.path.join(folder, x), os.listdir(folder))
    for pic_path in pic_list:
        print pic_path
        try:
            align_face(pic_path)
        except:
            print 'error :', pic_path
            continue

    pass