#-*- coding: utf-8 -*-
__author__ = 'liubo-it'


import os
import numpy as np
from scipy.misc import imread, imresize
import pdb
from time import time
import traceback
import argparse
from load_face_data import read_one_rgb_pic
import msgpack
from keras.utils import np_utils


class Data():
    def shuffle(self, data, label, *args, **kwargs):
        new_data = np.zeros(shape=data.shape)
        new_label = np.zeros(shape=label.shape)
        length = new_data.shape[0]
        shuffled_index = range(length)
        np.random.shuffle(shuffled_index)
        for index in range(len(shuffled_index)):
            new_data[index] = data[shuffled_index[index]]
            new_label[index] = label[shuffled_index[index]]
        return new_data, new_label

    def split(self, data, label, train_split, *args, **kwargs):
        all_data_num = data.shape[0]
        train_num = int(all_data_num * train_split)
        train_data = data[:train_num]
        valid_data = data[train_num:]
        train_label = label[:train_num]
        valid_label = label[train_num:]
        return train_data, train_label, valid_data, valid_label


class ImageData(Data):
    def __init__(self, init_args, *args, **kwargs):
        self.img_row = init_args.img_row
        self.img_col = init_args.img_col
        self.img_channel = init_args.img_channel
        self.train_list = init_args.train_path_list
        self.valid_list = init_args.valid_path_list
        self.func_args_dic = init_args.func_args_dic
        self.pic_shape = (self.img_row, self.img_col, self.img_channel)
        self.nb_classes = init_args.nb_classes

    def load_img_data(self, path_list):
        all_data = []
        all_label = []
        for path, label in path_list:
            pic_arr = read_one_rgb_pic(path, self.pic_shape, self.func_args_dic)
            if pic_arr != None:
                all_data.append(pic_arr)
                all_label.append(label)
        all_data = np.asarray(all_data) / 255.0
        all_label = np.asarray(all_label)
        return all_data, all_label

    def getData(self):
        train_data, train_label = self.load_img_data(self.train_list)
        valid_data, valid_label = self.load_img_data(self.valid_list)
        train_data = np.transpose(train_data, (0, 3, 1, 2))
        valid_data = np.transpose(valid_data, (0, 3, 1, 2))
        valid_label = np_utils.to_categorical(valid_label, self.nb_classes)
        train_label = np_utils.to_categorical(train_label, self.nb_classes)
        return train_data, train_label, valid_data, valid_label



class DataFactory():
    def __init__(self, init_args, *args, **kwargs):
        '''
            这些参数都通过all_args的相关文件进行配置,在train_model中传入
        '''

        self.img_row = init_args.img_row
        self.img_col = init_args.img_col
        self.img_channel = init_args.img_channel
        self.func_args_dic = init_args.func_args_dic
        self.pack_file = init_args.pack_file   # 解压之后,得到train_list, valid_list
        self.nb_classes = init_args.nb_classes

    def create_args(self):

        parser = argparse.ArgumentParser(description='load image data ')
        # [
        #       [pic_path, label], [pic_path, label], [pic_path, label], [pic_path, label]
        # ]
        train_list, valid_list = msgpack.load(open(self.pack_file, 'rb'))
        train_list = train_list[:1000]
        valid_list = valid_list[:1000]
        parser.add_argument('--train_path_list', type=list,
                            default=train_list, help='训练集图片路径的list')
        parser.add_argument('--valid_path_list', type=list,
                            default=valid_list, help='验证集图片路径的list')
        parser.add_argument('--img_row', type=int,
                            default=self.img_row, help='image row size')
        parser.add_argument('--img_col', type=int,
                            default=self.img_col, help='image row size')
        parser.add_argument('--img_channel', type=int,
                            default=self.img_channel, help=' image channel num')
        parser.add_argument('--func_args_dic', type=dict,
                            default=self.func_args_dic, help='每个图片的处理方式')
        parser.add_argument('--nb_classes', type=int,
                            default=self.nb_classes, help='分类个数')
        init_args = parser.parse_args()

        return init_args

    def getDataLoader(self):
        init_args = self.create_args()
        dataLoader = ImageData(init_args)
        return dataLoader


def test(pack_file):
    start = time()
    dataFactory = DataFactory(pack_file)
    dataLoader = dataFactory.getDataLoader()
    train_data, train_label, valid_data, valid_label = dataLoader.getData()
    end = time()
    print train_data.shape, train_label.shape, valid_data.shape, valid_label.shape, (end-start)


if __name__ == '__main__':
    test('/data/annotate_list.p')

