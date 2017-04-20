# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId.py
@time: 2016/7/27 15:39
@contact: ustb_liubo@qq.com
@annotation: base_args
"""

import argparse
import numpy as np
import os
import sys

data_folder = '/data/liubo/face/vgg_face_dataset/'
model_folder = '/data/liubo/face/vgg_face_dataset/model/'
nb_classes = 600


def CRPS(label, pred):
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


def get_basic_args(person_num, img_row, img_col, img_channel, data_num, pack_file, func_args_dic, nb_classes):
    values = {'person_num': person_num, 'img_row': img_row,
              'img_col': img_col, 'data_num': data_num,
              'model_name': os.path.split(sys.argv[0])[1][:-3]}
    prefix = '%(model_name)s.' \
             '%(person_num)d.%(img_row)d-%(img_col)d.%(data_num)d' % values
    model_file = os.path.join(model_folder, '%s.model' % prefix)
    weight_file = os.path.join(model_folder, '%s.weight' % prefix)

    parser = argparse.ArgumentParser(description='train kaggle ')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='the batch size')
    parser.add_argument('--patience', type=int,
                        default=20, help='the num of val_loss not dec')
    parser.add_argument('--model_file', type=str,
                        help='model_file_name')
    parser.add_argument('--weight_file', type=str,
                        help='weight_file_name')
    parser.add_argument('--nb_epoch', type=int,
                        default=100, help='the number of training epochs')
    parser.add_argument('--evaluate',
                        default=CRPS, help='evaluate method ')
    parser.add_argument('--data_augment', type=bool,
                        default=True, help='augment data or not')
    parser.add_argument('--pack_file', type=str,
                        default=pack_file, help='训练集和测试集的图片路径的列表')
    parser.add_argument('--img_row', type=int,
                        default=img_row, help='图片像素 行数')
    parser.add_argument('--img_col', type=int,
                        default=img_col, help='图片像素 列数')
    parser.add_argument('--img_channel', type=int,
                        default=img_channel, help='图片 色彩通道数')
    parser.add_argument('--func_args_dic', type=dict,
                        default=func_args_dic, help='对图片的处理函数')
    parser.add_argument('--nb_classes', type=int,
                        default=nb_classes, help='分类数')
    args = parser.parse_args()
    args.model_file = model_file
    args.weight_file = weight_file
    return args


if __name__ == '__main__':
    print type(CRPS)
