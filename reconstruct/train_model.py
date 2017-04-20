# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId.py
@time: 2016/7/27 15:39
@contact: ustb_liubo@qq.com
@annotation: train_model
"""

import util
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
import importlib
import csv
from conf import code_folder
import sys
from time import time
from DataProcess.load_data import DataFactory
sys.path.append(code_folder)
from logging.config import fileConfig
import logging
import numpy as np
import os
import pdb
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

# 新版的keras可以直接使用向量的方式,不存在Sequential和Graph的不同
# 所有的数据一次读入


class Model():
    def __init__(self, network_model, pack_file, args_file, *args, **kwargs):
        '''
            :param model_file: 保存一个模型
            :param data_label: 数据类型
            :param args_file: 保存该模型的一个配置
            :return:
        '''
        # 模型结构加上训练参数构成一次训练
        self.network_model = network_model
        self.pack_file = pack_file
        self.args_file = args_file


    def load_model(self, args):
        model = importlib.import_module(self.network_model).get_model(args)
        return model


    # 不同的任务可能会有不同的读数据的方法,这里DataFactory根据data_label调用不同的读数据的函数
    def load_data(self, args):
        start = time()
        dataFactory = DataFactory(args)
        dataLoader = dataFactory.getDataLoader()
        train_data, train_label, valid_data, valid_label = dataLoader.getData()
        end = time()
        print_list = [train_data.shape, train_label.shape, valid_data.shape, valid_label.shape, (end - start)]
        util.log_print(logger_error, print_list)
        return train_data, train_label, valid_data, valid_label


    def load_args(self):
        args = importlib.import_module(self.args_file).get_args()
        return args


    def train_valid_model(self, model, train_valid_data, args):
        '''
        :param model: 模型会保留网络结构,根据loss保留训练好的参数
        :param train_valid_data:
        :param args: evaluate = args.evaluate : 不同的函数可以有不同的评价函数, 这个会给出predict_prob, 然后计算loss
                 不同的任务可以需要修改ImageDataGenerator, 非图像可能使用别的数据增加方法
                [evaluate给出的值越小越好,如果是准确率,需要提前处理(1-acc)]
        :return:
        '''
        model_file = args.model_file
        weight_file = args.weight_file
        data_augment = args.data_augment
        nb_epoch = args.nb_epoch
        batch_size = args.batch_size
        X_train, Y_train, X_test, Y_test = train_valid_data
        evaluate = args.evaluate
        if not os.path.exists(model_file):
            util.save_model(model, model_file)
        if os.path.exists(weight_file):
            model.load_weights(weight_file)
        last_crps = float('inf')
        if not data_augment:
            for e in range(nb_epoch):
                print('-'*40)
                print('Epoch', e)
                print('-'*40)
                print('Training...')
                model.fit([X_train], [Y_train], validation_data=([X_test], [Y_test]),
                          batch_size=batch_size, nb_epoch=1, shuffle=True)
                Y_predict_batch = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
                test_loss = evaluate(Y_test, Y_predict_batch)
                Y_train_predict_batch = model.predict_proba(X_train, batch_size=batch_size, verbose=1)
                train_loss = evaluate(Y_train,Y_train_predict_batch)
                print ('train loss', train_loss, 'test loss', test_loss)
                if last_crps > test_loss:
                    model.save_weights(weight_file, overwrite=True)
                    print ('save model')
                    last_crps = test_loss
                else:
                    print ('not save model')
        else:
            # 不使用已知的图片打断,打乱图片的排列顺序
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            datagen.fit(X_train, augment=False)

            try:
                predict_y = model.predict(X_test, batch_size=batch_size)
                last_acc_mean = np.mean(np.argmax(predict_y, axis=1) == np.argmax(Y_test, axis=1))
            except:
                logger_error.error(''.join(map(str, [X_test.shape, Y_test.shape])))
                last_acc_mean = 0.0
            print 'last_acc_mean :', last_acc_mean

            for epoch_id in range(nb_epoch):
                print '-'*50
                print 'current Epoch :', epoch_id, ' all Epoch :', nb_epoch

                progbar = generic_utils.Progbar(X_train.shape[0])
                shuffle_list = range(X_train.shape[0])
                np.random.shuffle(shuffle_list)
                X_train = X_train[shuffle_list]
                Y_train = Y_train[shuffle_list]
                progbar_index = 0
                for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size):
                    loss = model.train_on_batch(X_batch, Y_batch)
                    progbar.add(X_batch.shape[0], values=[('train loss', loss)])
                    progbar_index += X_batch.shape[0]
                    if progbar_index > X_train.shape[0]:
                        break

                # 每个epoch进行一次valid
                print '-'*50
                try:
                    predict_y = model.predict(X_test, batch_size=batch_size)
                    acc_mean = np.mean(np.argmax(predict_y, axis=1) == np.argmax(Y_test, axis=1))
                except:
                    logger_error.error(''.join(map(str, [X_test.shape, Y_test.shape])))
                    continue
                if acc_mean > last_acc_mean:
                    print 'save weight', 'acc_mean :', acc_mean
                    last_acc_mean = acc_mean
                    model.save_weights(weight_file, overwrite=True)
                else:
                    print 'not save weight', 'acc_mean :', acc_mean
        return model


class DeepIdModel(Model):
    def deepid_model_train_valid(self):
        deepid_args = self.load_args()
        deepid_model = self.load_model(deepid_args)
        train_data, train_label, valid_data, valid_label = self.load_data(deepid_args)
        deepid_train_valid_data = (train_data, train_label, valid_data, valid_label)
        deepid_model = self.train_valid_model(deepid_model, deepid_train_valid_data, deepid_args)
        return deepid_model


def test_model(model_name):
    network_model ='all_model.%s'%model_name
    pack_file = '/data/annotate_list.p'
    args_file = 'all_args.%s'%model_name
    deepidModel = DeepIdModel(network_model, pack_file, args_file)
    deepidModel.deepid_model_train_valid()


if __name__ == '__main__':
    test_model('DeepId')
