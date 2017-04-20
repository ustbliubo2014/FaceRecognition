# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: recognize_server_v3.py
@time: 2016/11/17 18:21
@contact: ustb_liubo@qq.com
@annotation: recognize_server_v3 : 只负责识别
"""
import sys
sys.path.insert(0, '/home/liubo-it/FaceRecognization/')
import logging
from logging.config import fileConfig
import os
import time
from recog_util import get_current_day, get_current_time, read_one_rgb_pic, normalize_rgb_array
from DetectAndAlign.align_interface import align_face_rgb_array
import cv2
import pdb
import dlib
from keras.models import model_from_json
from keras.optimizers import Adam
import keras.backend as K
import tornado.ioloop
import tornado.web
import json
import base64
import zlib
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


size = 96
log_dir = 'recognition_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def load_model():
    model_file = '/data/liubo/face/annotate_face_model/small_cnn.model'
    weight_file = '/data/liubo/face/annotate_face_model/small_cnn.weight'
    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load weights'
        model.load_weights(weight_file)
        get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                         [model.layers[-3].get_output_at(False)])
        return model, get_Conv_FeatureMap


def extract_feature(face_array):
    '''
    :param face_array: RGB读入的数据(dtype=uint8) [0-255]
    :return:
    '''
    start = time.time()
    current_day = get_current_day()
    current_time = get_current_time()
    log_file = open(os.path.join(log_dir, current_day + '.txt'), 'a')
    face_align_array = align_face_rgb_array(face_array, bb=dlib.rectangle(0, 0, face_array.shape[1], face_array.shape[0]))
    face_align_array = normalize_rgb_array(cv2.resize(face_align_array, (size, size)))
    feature_vector = get_Conv_FeatureMap([face_align_array, 0])[0].copy()
    log_file.write('\t'.join(map(str, [current_time, 'recognition_time :', (time.time() - start)])) + '\n')
    log_file.close()
    print (time.time() - start)
    return feature_vector


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        request_type = self.get_body_argument('request_type')
        if request_type == 'recognization':
            try:
                face_array_str = self.get_body_argument("image")
                image = base64.decodestring(face_array_str)
                image = zlib.decompress(image)
                face_array = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1)
                feature_vector = extract_feature(face_array)
                feature_vector_str = ','.join(map(str, feature_vector[0]))
                self.write(feature_vector_str)
            except:
                return


if __name__ == '__main__':
    model, get_Conv_FeatureMap = load_model()
    port = 7777
    application = tornado.web.Application([(r"/", MainHandler), ])
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

