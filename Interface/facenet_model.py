# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: facenet_model.py
@time: 2017/2/8 15:22
@contact: ustb_liubo@qq.com
@annotation: facenet_model
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


sess = tf.Session()
meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(recognize_model_dir))
saver = tf.train.import_meta_graph(os.path.join(recognize_model_dir, meta_file))
saver.restore(sess, os.path.join(recognize_model_dir, ckpt_file))
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


def extract_feature_from_file(pic_path):
    # 包含人脸检测和人脸识别
    # align_face(pic_path)
    images = facenet.load_data([pic_path], False, False, image_size)
    if images.shape[-1] != 3:
        return None
    feed_dict = {images_placeholder: images}
    face_feature = sess.run(embeddings, feed_dict=feed_dict)
    return face_feature


if __name__ == '__main__':
    pic_path = sys.argv[1]
    face_feature = extract_feature_from_file(pic_path)
