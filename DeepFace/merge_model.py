# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: merge_model.py
@time: 2016/8/8 16:53
@contact: ustb_liubo@qq.com
@annotation: merge_model : 将DeepFace的卷积层和新训练的softmax拼接到一起(新建一个模型,将两个就的模型的参数复制过去)
"""

import sys
import logging
from logging.config import fileConfig
import os
from keras.models import model_from_json
from keras.optimizers import Adam
import pdb
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

def exchange_model_weight(cnn_model, mlp_model, merge_model):
    '''
        将cnn_model的softmax层换成softmax_model
    :param cnn_model: Conv+Dense+softmax
    :param mlp_model: mlp
    :param merge_model: 合并后的网络
    :return:
    '''
    for i, layer in enumerate(cnn_model.layers[0:-1]):
        if layer.__class__ in [Convolution2D, Dense]:
            merge_model.layers[i].set_weights(cnn_model.layers[i].get_weights())
    for j, layer in enumerate(mlp_model.layers):
        if layer.__class__ in [Dense]:
            merge_model.layers[i+j].set_weights(mlp_model.layers[j].get_weights())


def load_model(model_file, weight_file):
    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load_weights'
        model.load_weights(weight_file)
        return model


if __name__ == '__main__':
    nb_class = 737
    cnn_model_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.model'
    cnn_weight_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.weight'
    # softmax_model_file = '/data/liubo/face/vgg_face_dataset/model/annotate_softmax_classify.model'
    # softmax_weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate_softmax_classify.weight'
    mlp_model_file = '/data/liubo/face/vgg_face_dataset/model/annotate_mlp_classify_%d.model'%nb_class
    mlp_weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate_mlp_classify_%d.weight'%nb_class
    merge_model_file = '/data/liubo/face/vgg_face_dataset/model/annotate_deep_face_%d.model'%nb_class
    merge_weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate_deep_face_%d.weight'%nb_class
    print merge_weight_file, merge_model_file
    cnn_model = model_from_json(open(cnn_model_file, 'r').read())
    cnn_model.load_weights(cnn_weight_file)
    mlp_model = model_from_json(open(mlp_model_file, 'r').read())
    mlp_model.load_weights(mlp_weight_file)
    merge_model = model_from_json(open(merge_model_file, 'r').read())
    merge_model.load_weights(merge_weight_file)
    exchange_model_weight(cnn_model, mlp_model, merge_model)
    merge_model.save_weights(merge_weight_file, overwrite=True)
    merge_model.summary()
