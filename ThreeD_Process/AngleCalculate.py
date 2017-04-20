# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: AngleCalculate.py
@time: 2016/9/21 15:10
@contact: ustb_liubo@qq.com
@annotation: AngelCalculate
"""
import sys
import os
import logging
from logging.config import fileConfig
import _init_paths
import caffe
# import cv2
from scipy.misc import imresize, imread
import dlib
import numpy as np
from time import time
import pdb
from CriticalPointDetection import CriticalPointDetection
import traceback
import sys
sys.path.append('/home/liubo-it/FaceRecognization/')
from DetectAndAlign.align_interface import align_face
import shutil

reload(sys)
sys.setdefaultencoding("utf-8")


class AngleCalculate(object):
    def __init__(self):
        self.model_folder = '/data/liubo/face/model/face_landmark/'
        self.vgg_point_MODEL_FILE = os.path.join(self.model_folder, 'deploy.prototxt')
        self.vgg_point_PRETRAINED = os.path.join(self.model_folder, '68point_dlib_with_pose.caffemodel')
        self.mean_filename = os.path.join(self.model_folder, 'VGG_mean.binaryproto')
        self.vgg_height = 224
        self.vgg_width = 224
        self.channel_num = 3
        self.pose_blobName = 'poselayer'
        self.critical_point_blobName = '68point'
        self.mean = None
        caffe.set_mode_cpu()
        # caffe.set_device(1)
        self.criticalPointDetection = CriticalPointDetection()


    def load_model(self):
        self.vgg_point_net = caffe.Net(self.vgg_point_MODEL_FILE, self.vgg_point_PRETRAINED, caffe.TEST)
        proto_data = open(self.mean_filename, "rb").read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        self.mean = caffe.io.blobproto_to_array(a)[0]
        print 'load model'


    def get_cut_size(self, bbox, left, right, top, bottom):
        # left, right, top, and bottom
        box_width = bbox[1] - bbox[0]
        box_height = bbox[3] - bbox[2]
        cut_size = np.zeros((4))
        cut_size[0] = bbox[0] + left * box_width
        cut_size[1] = bbox[1] + (right - 1) * box_width
        cut_size[2] = bbox[2] + top * box_height
        cut_size[3] = bbox[3] + (bottom - 1) * box_height
        return cut_size


    def revise_box(self, img, bbox):
        img_height = img.shape[0] - 1
        img_width = img.shape[1] - 1
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[2] < 0:
            bbox[2] = 0
        if bbox[3] < 0:
            bbox[3] = 0
        if bbox[0] > img_width:
            bbox[0] = img_width
        if bbox[1] > img_width:
            bbox[1] = img_width
        if bbox[2] > img_height:
            bbox[2] = img_height
        if bbox[3] > img_height:
            bbox[3] = img_height
        return bbox


    def get_rgb_test_part(self, bbox, left, right, top, bottom, img, height, width):
        face_box = self.get_cut_size(bbox, left, right, top, bottom)
        revise_box = self.revise_box(img, face_box)
        revise_box = np.asarray(revise_box, dtype=np.int)
        face = img[revise_box[2]: revise_box[3], revise_box[0]: revise_box[1], :]
        face = imresize(face, (height, width, self.channel_num))
        face = face.astype('float32')
        return face


    def cal_angle(self, face_img):
        '''
        :param face_img: 检测的的人脸图片
        :return: predict_points(关键点)
                pose_predict(姿势) : [[pitch, yaw, roll]]  (Pitch : 俯仰;  Yaw : 摇摆; Roll : 倾斜)
        '''
        try:
            faces = np.zeros((1, 3, self.vgg_height, self.vgg_width))
            # 提前做人脸检测和人脸对齐, 不需要在这里在做一次人脸检测,浪费时间
            # face_img = cv2.resize(face_img, (self.vgg_height, self.vgg_width), interpolation=cv2.INTER_AREA)
            face_img = imresize(face_img, (self.vgg_height, self.vgg_width, self.channel_num))
            normal_face = np.zeros(self.mean.shape)
            normal_face[0] = face_img[:, :, 0]
            normal_face[1] = face_img[:, :, 1]
            normal_face[2] = face_img[:, :, 2]
            normal_face = normal_face - self.mean
            faces[0] = normal_face
            # faces.shape : (1, 3, 224, 224) [batch_size = 1的测试集

            data4DL = np.zeros([faces.shape[0], 1, 1, 1])
            self.vgg_point_net.set_input_arrays(faces.astype(np.float32), data4DL.astype(np.float32))
            self.vgg_point_net.forward()
            predict_points = self.vgg_point_net.blobs[self.critical_point_blobName].data[0]

            pose_predict = self.vgg_point_net.blobs[self.pose_blobName].data
            pose_predict = pose_predict * 50

            return predict_points, pose_predict
        except:
            traceback.print_exc()
            return None, None


def angle_filter(src_folder, dst_folder):
    angle_threshold = 10
    angle_calculate = AngleCalculate()
    angle_calculate.load_model()
    # folder = 'test_cluster'
    pic_list = os.listdir(src_folder)
    for pic in pic_list:
        src_pic_path = os.path.join(src_folder, pic)
        start = time()
        face_img = imread(src_pic_path)
        face_img = face_img[:, :, :3]
        predict_points, pose_predict = angle_calculate.cal_angle(face_img)
        end = time()
        if abs(pose_predict[0][0]) > angle_calculate or abs(pose_predict[0][1]) > angle_threshold or \
            abs(pose_predict[0][2]) > angle_threshold:
            os.remove(src_pic_path)
            continue
        else:
            dst_pic_path = os.path.join(dst_folder, pic)
            shutil.move(src_pic_path, dst_pic_path)
        print pic, pose_predict, (end - start)


def detect_filter():
    # 用opencv检测, 得不到的结果直接删掉
    folder = 'test_cluster_angle'
    pic_list = os.listdir(folder)
    for pic in pic_list:
        pic_path = os.path.join(folder, pic)
        face = align_face(pic_path)
        if face == None:
            os.remove(pic_path)


if __name__ == '__main__':
    pass
    angle_filter(src_folder='test_cluster_1000000', dst_folder='cluster_pic')
    # detect_filter()

