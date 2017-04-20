# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: 3d_trans.py
@time: 2016/9/23 16:02
@contact: ustb_liubo@qq.com
@annotation: 3d_trans
"""
import sys
import logging
from logging.config import fileConfig
import os
import dlib
from build_3d import utils
from build_3d import models
import cv2
from build_3d import FaceRendering
from build_3d import NonLinearLeastSquares

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


class Trans_3D(object):
    def __init__(self):
        # self.predictor_path = "/data/liubo/face/model/shape_predictor_68_face_landmarks.dat"
        # self.FaceModel_file = '/data/liubo/face/model/3d_model/candide.npz'
        self.predictor_path = "D:/data/face/DlibProject-master/shape_predictor_68_face_landmarks.dat"
        self.FaceModel_file = 'E:/git/FaceSwap-master/candide.npz'
        self.width = 224
        self.height = 224


    def load_model(self):
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.mean3DShape, self.blendshapes, self.mesh, self.idxs3D, self.idxs2D = utils.load3DFaceModel(self.FaceModel_file)
        self.projectionModel = models.OrthographicProjectionBlendshapes(self.blendshapes.shape[0])
        self.renderer = FaceRendering.FaceRenderer(width=self.width, height=self.height)
        self.projectionModel = models.OrthographicProjectionBlendshapes(self.blendshapes.shape[0])


    def trans_3d(self, profile_face_img, front_face_img):
        '''
        :param profile_face_img: 检测到的人脸图片
            如果角度比较小,不用转换,只有角度大时才进行转换
        :param front_face_img: 正脸图片, 将侧脸图片转换成正脸的角度
        :return:
        '''
        profile_face_coords = utils.getFaceTextureCoords(profile_face_img, self.mean3DShape, self.blendshapes,
                                            self.idxs2D, self.idxs3D, self.detector, self.predictor)
        self.renderer.load_img(front_face_img, profile_face_img, profile_face_coords, self.mesh)
        shapes2D = utils.getFaceKeypoints(front_face_img, self.detector, self.predictor)
        if shapes2D:
            shape2D = shapes2D[0]
            modelParams = self.projectionModel.getInitialParameters(
                self.mean3DShape[:, self.idxs3D], shape2D[:, self.idxs2D])
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, self.projectionModel.residual,
                                            self.projectionModel.jacobian,
                                            ([self.mean3DShape[:, self.idxs3D], self.blendshapes[:, :, self.idxs3D]],
                                            shape2D[:, self.idxs2D]), verbose=0)
            shape3D = utils.getShape3D(self.mean3DShape, self.blendshapes, modelParams)
            renderedImg = self.renderer.render(shape3D)
            return renderedImg


def trans_3d(trans_3d, front_path, raw_path, dst_path):
    print raw_path
    front_face_img = cv2.resize(cv2.imread(front_path), (224, 224))
    profile_face_img = cv2.resize(raw_path, (224, 224))
    trans_img = trans_3d.trans_3d(profile_face_img, front_face_img)
    cv2.imwrite(dst_path, trans_img)

if __name__ == '__main__':
    trans = Trans_3D()
    trans.load_model()


    # raw_folder = 'lfw_face'
    # dst_folder = 'lfw_face_3d'
    # front_pic_path = '/data/liubo/face/lfw_face/Zhu_Rongji/Zhu_Rongji_0008.jpg'
    # person_list = os.listdir(raw_folder)
    # if not os.path.exists(dst_folder):
    #     os.makedirs(dst_folder)
    # for person in person_list:
    #     raw_person_path = os.path.join(raw_folder, person)
    #     dst_person_path = os.path.join(dst_folder, person)
    #     if not os.path.exists(dst_person_path):
    #         os.makedirs(dst_person_path)
    #     pic_list = os.listdir(raw_person_path)
    #     for pic in pic_list:
    #         raw_pic_path = os.path.join(raw_person_path, pic)
    #         dst_pic_path = os.path.join(dst_person_path, pic)
    #         trans_3d(trans, front_pic_path, raw_pic_path, dst_pic_path)


    profile_face = 'xiejunping1468293608.52.png_face_0.jpg'
    front_face = '1476442182.77.png'
    profile_face_img = cv2.imread(profile_face)
    front_face_img = cv2.imread(front_face)
    front_face_img = cv2.resize(front_face_img, (224, 224))
    profile_face_img = cv2.resize(profile_face_img, (224, 224))
    trans_img = trans.trans_3d(profile_face_img, front_face_img)
    cv2.imwrite('trans_img2.jpg', trans_img)
