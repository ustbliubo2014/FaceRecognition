# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_annotate_batch.py
@time: 2016/8/3 10:24
@contact: ustb_liubo@qq.com
@annotation: DeepId_annotate_batch
"""
import os
from DeepID.DeepId1_batch.DeepId1_batch import train
from DeepID.util.DeepId import writer
from time import time
from Queue import Queue
import threading
import msgpack
from DeepID.util.MyThread import MyThread
from optparse import OptionParser
import pdb
import sys
from logging.config import fileConfig
import logging

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


class DeepIdBatch(object):
    def __init__(self,  maxsize=3, epoch_num=100):
        # 每个batch数据很大,读入后可以多训练几轮
        self.train_queue = Queue(maxsize)
        self.valid_queue = Queue(maxsize)
        self.sample_list_file = '/data/annotate_list.p'
        self.model_folder = '/data/liubo/face/vgg_face_dataset/model'
        self.train_batch_size = 1280
        self.valid_batch_size = 128
        self.epoch_num = epoch_num
        self.pic_shape = (128, 128, 3)
        self.input_shape = (3, 128, 128)
        self.model_file = os.path.join(self.model_folder, 'annotate.all_data.mean.small.rgb.deepid_relu.deep_filters.batch.model')
        self.weight_file = os.path.join(self.model_folder,'annotate.all_data.mean.small.rgb.deepid_relu.deep_filters.batch.weight')

    def read_train(self):
        train_sample_list, valid_sample_list = msgpack.load(open(self.sample_list_file, 'rb'))
        person_num = len(set([tmp[1] for tmp in train_sample_list]))
        train_batch_num = len(train_sample_list) / self.train_batch_size
        valid_batch_num = len(valid_sample_list) / self.valid_batch_size
        train_valid_model_args = (self.input_shape, person_num, self.model_file, self.weight_file, self.train_queue,
                    self.valid_queue, self.epoch_num, train_batch_num, valid_batch_num,)
        train_write_args = (self.train_queue, self.epoch_num, self.train_batch_size, train_sample_list,
                            train_batch_num, person_num, self.pic_shape)
        valid_write_args = (self.valid_queue, self.epoch_num, self.valid_batch_size, valid_sample_list,
                            valid_batch_num, person_num, self.pic_shape)
        train_valid_model_thread = MyThread(func=train, args=train_valid_model_args, name='train_valid_model')
        train_write_thread = MyThread(func=writer, args=train_write_args, name='train_write')
        valid_write_thread = MyThread(func=writer, args=valid_write_args, name='valid_write')
        train_valid_model_thread.start()
        train_write_thread.start()
        valid_write_thread.start()
        train_valid_model_thread.join()
        train_write_thread.join()
        valid_write_thread.join()


if __name__ == '__main__':
    deepid = DeepIdBatch()
    deepid.read_train()


