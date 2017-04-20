# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: detect_recognize_server.py
@time: 2016/11/24 17:05
@contact: ustb_liubo@qq.com
@annotation: detect_recognize_server
    使用自己的检测,  返回的是检测到的人脸, 人脸部分检测的时候用大一的图片,返回时返回正好的图片
    增加角度过滤, 角度不满足的不在返回特征(返回检测结果)[返回全0的feature]
"""
import sys
sys.path.insert(0, '/home/liubo-it/FaceRecognization/')
import logging
from logging.config import fileConfig
import os
import time
from recog_util import get_current_day, get_current_time, normalize_rgb_array
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
import numpy as np
import requests
from PIL import Image
from StringIO import StringIO
import msgpack_numpy
import traceback
import math
from recog_util import is_blur, blur_threshold
from scipy.misc import imsave

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


size = 96
pitch_threshold = 15
yaw_threshold = 20
roll_threshold = 20
feature_dim = 512
detect_url = "http://10.160.164.25:9999/"
angle_url = "http://10.160.164.26:7788/"
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
        print 'load finish'
        return model, get_Conv_FeatureMap


def check_face_img(face_img):
    # pose_predict(姿势): [[pitch, yaw, roll]](Pitch: 俯仰; Yaw: 摇摆; Roll: 倾斜)
    '''
    :param face_img: 人脸对应的矩阵
    :param image_id: 图片id
    :return: 是否进行识别(False:不进行识别)
    '''

    current_day = get_current_day()
    log_file = open(os.path.join(log_dir, current_day + '.txt'), 'a')

    face_img_str = base64.b64encode(msgpack_numpy.dumps(face_img))
    request = {"request_type": 'check_pose', "face_img_str": face_img_str, "image_id": str(time.time())}
    result = requests.post(angle_url, data=request)

    try:
        if result.status_code == 200:
            pose_predict = json.loads(result.content)["pose_predict"]
            if not pose_predict:  # 加载失败
                log_file.write('\t'.join(map(str, ['pose filter request'])) + '\n')
                log_file.close()
                return False
            else:
                pose_predict = msgpack_numpy.loads(base64.b64decode(pose_predict))
                if pose_predict == None:
                    log_file.write('\t'.join(map(str, ['pose filter detect'])) + '\n')
                    log_file.close()
                    return False
                pitch, yaw, roll = pose_predict[0]
                if math.fabs(pitch) < pitch_threshold and math.fabs(yaw) < yaw_threshold and math.fabs(roll) < roll_threshold:
                    log_file.write('\t'.join(map(str, ['pose not filter', str(pose_predict[0])])) + '\n')
                    log_file.close()
                    return True
                else:
                    log_file.write('\t'.join(map(str, ['pose filter threshold', str(pose_predict[0])])) + '\n')
                    log_file.close()
                    return False
        else:
            return False
    except:
        traceback.print_exc()
        log_file.close()
        return False


def detect_recognize_face(raw_img):
    # 自己检测时使用大一点的图片, 返回的是正好的图片
    # 检测不在需要resize
    try:
        request = {
            "image_id": "czc_test",
            # "image": base64.encodestring(((cv2.imencode('.jpg', small_img)[1].tostring())))
            "image": base64.encodestring(((cv2.imencode('.jpg', raw_img)[1].tostring())))
        }
        result = requests.post(detect_url, data=request)
        face_pos = json.loads(result.content)['detection']
        face_num = len(face_pos)
        result = ['faces num: '+str(face_num)]
        for index in range(len(face_pos)):
            try:
                x = int(face_pos[index][0])
                w = int(face_pos[index][2])
                y = int(face_pos[index][1])
                h = int(face_pos[index][3])
                center_x = x + w / 2
                center_y = y + h / 2
                # 用于识别的图片
                new_x_min = int(max(center_x - w * 0.85, 0))
                new_x_max = int(min(center_x + w * 0.85, raw_img.shape[1]))
                new_y_min = int(max(center_y - h * 0.85, 0))
                new_y_max = int(min(center_y + h * 0.85, raw_img.shape[0]))
                face_array = raw_img[new_y_min:new_y_max, new_x_min:new_x_max]
                face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
                feature_vector = extract_feature(face_array)
                result.append(','.join(map(str, [x, y, w, h])))
                result.append(','.join(map(str, feature_vector[0]))+',')   # 和研究院的输出保持一致
            except:
                traceback.print_exc()
                continue
        return '\n'.join(result)
    except:
        traceback.print_exc()
        return ''


def extract_feature(face_array):
    '''
        先计算角度, 满足条件后再进行对齐和识别
    :param face_array: RGB读入的数据(dtype=uint8) [0-255]
    :return:
    '''
    start = time.time()
    current_day = get_current_day()
    current_time = get_current_time()
    log_file = open(os.path.join(log_dir, current_day + '.txt'), 'a')

    blur_sign, blur_var = is_blur(face_array)
    if blur_sign:
        print 'blur_filter', blur_var
        feature_vector = np.reshape(np.asarray([0] * feature_dim), (1, feature_dim))
        log_file.write('\t'.join(map(str, [current_time, 'blur', 'not_process', 'recognition_time :',
                                           (time.time() - start)])) + '\n')
        return feature_vector

    need_process = check_face_img(face_img=face_array)
    if not need_process:
        feature_vector = np.reshape(np.asarray([0] * feature_dim), (1, feature_dim))
        log_file.write('\t'.join(map(str, [current_time, 'pose', 'not_process', 'recognition_time :',
                                           (time.time() - start)])) + '\n')
        return feature_vector

    # 自己检测对齐
    face_align_array = align_face_rgb_array(face_array, bb=None)
    face_align_array = normalize_rgb_array(cv2.resize(face_align_array, (size, size)))
    feature_vector = get_Conv_FeatureMap([face_align_array, 0])[0].copy()
    log_file.write('\t'.join(map(str, [current_time, 'process', 'recognition_time :', (time.time() - start)])) + '\n')
    log_file.close()
    return feature_vector


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        start = time.time()
        current_day = get_current_day()
        current_time = get_current_time()
        log_file = open(os.path.join(log_dir, current_day + '.txt'), 'a')
        pic_binary_data = self.request.body
        open('/tmp/face_recog_tmp/'+str(time.time())+'.jpg', 'wb').write(pic_binary_data)
        img_buffer = StringIO(pic_binary_data)
        img_array = np.array(Image.open(img_buffer))
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        print 'img_array_bgr.shape :', img_array_bgr.shape
        result = detect_recognize_face(img_array_bgr)
        self.write(result)
        end = time.time()
        log_file.write('\t'.join(map(str, [current_time, 'get_pic', 'all_time :', (end - start)])) + '\n')
        log_file.close()


if __name__ == '__main__':
    model, get_Conv_FeatureMap = load_model()
    port = 7777 
    application = tornado.web.Application([(r"/", MainHandler), ])
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()


