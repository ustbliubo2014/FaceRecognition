# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_batch_model.py
@time: 2016/7/4 16:54
@contact: ustb_liubo@qq.com
@annotation: DeepId_batch_model
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='DeepId_batch_model.log',
                    filemode='w')

import numpy as np
import pdb
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model, model_from_json, Graph
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.utils import generic_utils
import os
from util_model import create_deepId_network
from util_model import build_deepid_model
from keras import backend as K


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
    :return: [数据旋转,平移等操作在读数据的时候完成]
    '''
    print 'model_file :', model_file
    print 'weight_file :', weight_file
    if os.path.exists(model_file):
        print 'load model'
        model = model_from_json(open(model_file).read())
        # opt = SGD(lr=0.01)
        opt = Adam()
        # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    else:
        model = build_deepid_model(create_deepId_network, input_shape, nb_classes)
        open(model_file,'w').write(model.to_json())
    if os.path.exists(weight_file):
        print 'load weight'
        model.load_weights(weight_file)

    last_value_loss_mean = float('inf')
    for epoch_id in range(epoch_num):
        print '-'*50
        print 'current Epoch :', epoch_id, ' all Epoch :', epoch_num
        progbar = generic_utils.Progbar(train_batch_num)
        # train
        for batch_id in range(train_batch_num):
            batch_x, batch_y = train_queue.get()
            # print batch_x.shape, batch_y.shape
            loss = model.train_on_batch(batch_x, batch_y)
            progbar.add(1, values=[('train loss', loss)])
        # valid
        print '-'*50
        progbar = generic_utils.Progbar(valid_batch_num)
        for batch_id in range(valid_batch_num):
            batch_x, batch_y = valid_queue.get()
            loss = model.test_on_batch(batch_x, batch_y)#直接得到loss
            progbar.add(1, values=[('value loss', loss)])
        value_loss_mean = progbar.sum_values.get('value loss')[1]
        if value_loss_mean < last_value_loss_mean:
            last_value_loss_mean = value_loss_mean
            model.save_weights(weight_file, overwrite=True)



def load_deepid_model(model_file, weight_file):
    print 'model_file :', model_file
    print 'weight_file :', weight_file
    model = model_from_json(open(model_file, 'r').read())
    model.load_weights(weight_file)
    get_Conv_FeatureMap = K.function([model.layers[1].layers[0].get_input_at(False), K.learning_phase()],
                                     [model.layers[1].layers[-1].get_output_at(False)])
    return model, get_Conv_FeatureMap


