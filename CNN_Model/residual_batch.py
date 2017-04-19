# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: residual_batch.py
@time: 2016/7/19 15:59
@contact: ustb_liubo@qq.com
@annotation: residual_batch
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='residual_batch.log',
                    filemode='a+')

from time import time
from keras import backend as K
import os
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad
)
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from residual import resnet

def train(input_shape, nb_classes, model_file, weight_file, train_queue, valid_queue, epoch_num,
          train_batch_num, valid_batch_num):
    '''
    :param input_shape: 图片的shape[1:]
    :param nb_classes: person_num
    :param model_file:
    :param weight_file:
    :param train_queue: 存放训练数据
    :param valid_queue: 存放验证数据
    :param epoch_num:
    :param train_batch_num: epoch_num+batch_num用于判断读取多少次queue
    :param valid_batch_num:
    :return:
    '''
    if os.path.exists(model_file):
        print 'load model'
        model = model_from_json(open(model_file).read())
    else:
        model = resnet(input_shape, nb_classes)
        open(model_file,'w').write(model.to_json())
    opt = Adagrad()
    # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    if os.path.exists(weight_file):
        print 'load weight'
        model.load_weights(weight_file)


    progbar = generic_utils.Progbar(valid_batch_num)
    for batch_id in range(valid_batch_num):
        batch_x, batch_y = valid_queue.get()
        predict_y = model.predict_on_batch(batch_x)
        acc = np.mean(np.argmax(predict_y,axis=1) == np.argmax(batch_y, axis=1))
        progbar.add(1, values=[('acc', acc)])
    acc_mean = progbar.sum_values.get('acc')[0] / progbar.sum_values.get('acc')[1]
    last_acc_mean = acc_mean


    for epoch_id in range(epoch_num):
        print '-'*50
        print 'current Epoch :', epoch_id, ' all Epoch :', epoch_num
        progbar = generic_utils.Progbar(train_batch_num)
        for batch_id in range(train_batch_num):
            batch_x, batch_y = train_queue.get()

            loss = model.train_on_batch(batch_x, batch_y)
            progbar.add(1, values=[('train loss', loss)])
        # valid
        print '-'*50
        progbar = generic_utils.Progbar(valid_batch_num)
        for batch_id in range(valid_batch_num):
            batch_x, batch_y = valid_queue.get()
            predict_y = model.predict_on_batch(batch_x)
            acc = np.mean(np.argmax(predict_y, axis=1) == np.argmax(batch_y, axis=1))
            progbar.add(1, values=[('acc', acc)])
        acc_mean = progbar.sum_values.get('acc')[0] / progbar.sum_values.get('acc')[1]
        if acc_mean > last_acc_mean:
            print 'save weight'
            last_acc_mean = acc_mean
            model.save_weights(weight_file, overwrite=True)


