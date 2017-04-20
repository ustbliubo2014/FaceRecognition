# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: caffe2keras.py
@time: 2016/8/8 11:44
@contact: ustb_liubo@qq.com
@annotation: caffe2keras : 先根据caffe的proto设置keras的model,然后读入keras的模型,加载caffe的参数,最后保存keras的model和weight
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
import caffe
from deepface import deep_face
import pdb
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


class Model2Keras(object):
    def __init__(self, kerasmodel=None, prototxt=None, caffemodel=None):
        if not kerasmodel or not prototxt or not caffemodel:
            raise RuntimeError('kerasmodel and protext and caffemodel must specified')

        self.kerasmodel = kerasmodel
        self.prototxt = prototxt
        self.caffemodel = caffemodel

    def load_caffe_params(self):
        net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

        weights_layers = []
        params = net.params.values()
        for i, layer in enumerate(self.kerasmodel.layers):
            if layer.__class__ in [Convolution2D, Dense]:
                weights_layers.append(layer)

        if len(weights_layers) == len(params):
            for i in range(len(weights_layers)):
                print 'model shapes: ', params[i][0].data.shape, 'and', params[i][1].data.shape
                if weights_layers[i].__class__ == Dense:
                    print 'load Dense weight'
                    weights_layers[i].set_weights([params[i][0].data.T, params[i][1].data])
                elif weights_layers[i].__class__ == Convolution2D:
                # if weights_layers[i].__class__ == Convolution2D:
                    w_weights = params[i][0].data
                    print 'load Convolution2D weight'
                    for w_i in xrange(w_weights.shape[0]):
                        for w_j in xrange(w_weights.shape[1]):
                            w_weights[w_i][w_j] = np.flipud(np.fliplr(w_weights[w_i][w_j]))
                    weights_layers[i].set_weights([w_weights, params[i][1].data])

    def save_model(self, model_file, weight_file):
        print 'save model'
        self.kerasmodel.save_weights(weight_file, overwrite=True)
        print 'save weight'
        open(model_file, 'w').write(self.kerasmodel.to_json())


if __name__ == '__main__':
    keras_model = deep_face(input_shape=(3, 224, 224), nb_classes=2622)
    caffe_proto_file = '/home/liubo-it/VGGFaceModel-master/VGG_FACE_deploy.prototxt'
    caffe_model_file = '/home/liubo-it/VGGFaceModel-master/VGG_FACE.caffemodel'
    keras_model_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.model'
    keras_weight_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.weight'
    model_trans = Model2Keras(kerasmodel=keras_model, prototxt=caffe_proto_file, caffemodel=caffe_model_file)
    model_trans.load_caffe_params()
    model_trans.save_model(keras_model_file, keras_weight_file)

