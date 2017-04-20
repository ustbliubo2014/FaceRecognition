# encoding: utf-8

import numpy as np
import cv2
import models
import NonLinearLeastSquares
import pdb
from time import time
import dlib

def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ

def flipWinding(triangle):
    return [triangle[1], triangle[0], triangle[2]]

def fixMeshWinding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = getNormal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flipWinding(triangle)
    return mesh

def getShape3D(mean3DShape, blendshapes, params):
    # skalowanie
    s = params[0]
    # rotacja
    r = params[1:4]
    #przesuniecie (translacja)
    t = params[4:6]
    w = params[6:]

    #macierz rotacji z wektora rotacji, wzor Rodriguesa
    # 罗德里格斯旋转公式(用于3维矩阵或向量的旋转)
    # r:(1x3) ; R(3x3)
    R = cv2.Rodrigues(r)[0]
    # 摄像头中的图片加上平均的3D图片
    shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D


def getMask(renderedImg):
    mask = np.zeros(renderedImg.shape[:2], dtype=np.uint8)

def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    mean3DShape = faceModelFile["mean3DShape"]
    mesh = faceModelFile["mesh"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    blendshapes = faceModelFile["blendshapes"]
    mesh = fixMeshWinding(mesh, mean3DShape)

    return mean3DShape, blendshapes, mesh, idxs3D, idxs2D

def getFaceKeypoints(face_img, detector, predictor):
    '''
    :param face_img: 一个人的人脸图片
    :param detector: dlib的人脸检测(直接使用原始的人脸图片, 关键点不能拟合)
    :param predictor: dlib的关键点检测
    :return:
    '''
    # shape2D =  np.array([[p.x, p.y] for p in predictor(
    #     face_img, dlib.rectangle(0, 0, face_img.shape[0], face_img.shape[1])).parts()])
    # return [shape2D.T]

    dets = detector(face_img, 1)
    if len(dets) == 0:
        return None
    shapes2D = []
    for det in dets:
        faceRectangle = det
        dlibShape = predictor(face_img, faceRectangle)
        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        shape2D = shape2D.T
        shapes2D.append(shape2D)
    return shapes2D


def getFaceTextureCoords(img, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor):
    projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])
    keypoints = getFaceKeypoints(img, detector, predictor)[0]
    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian,
                                                ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]),
                                                verbose=0)
    textureCoords = projectionModel.fun([mean3DShape, blendshapes], modelParams)
    return textureCoords
