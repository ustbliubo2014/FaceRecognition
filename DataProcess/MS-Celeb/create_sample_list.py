# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: create_sample_list.py
@time: 2016/9/6 14:56
@contact: ustb_liubo@qq.com
@annotation: create_sample_list
"""
import sys
import logging
from logging.config import fileConfig
import os
import pdb
import msgpack_numpy
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")


def create_sample_list(root_folder, train_valid_sample_list_file, verif_sample_list_file):
    person_list = os.listdir(root_folder)
    train_valid_sample_list = []
    verif_sample_list = []
    # 留下2000人用于人脸验证模型
    for person_index, person_name in enumerate(person_list[:-2000]):
        person_path = os.path.join(root_folder, person_name)
        pic_list = map(lambda y: (y, person_index), map(lambda x: os.path.join(person_path, x), os.listdir(person_path)))
        train_valid_sample_list.extend(pic_list)
    np.random.shuffle(train_valid_sample_list)
    train_num = int(len(train_valid_sample_list) * 0.8)
    train_sample_list = train_valid_sample_list[:train_num]
    valid_sample_list = train_valid_sample_list[train_num:]
    msgpack_numpy.dump((train_sample_list, valid_sample_list), open(train_valid_sample_list_file, 'wb'))
    for person_index, person_name in enumerate(person_list[-2000:]):
        person_path = os.path.join(root_folder, person_name)
        pic_list = map(lambda y: (y, person_index), map(lambda x: os.path.join(person_path, x), os.listdir(person_path)))
        verif_sample_list.extend(pic_list)
    msgpack_numpy.dump(verif_sample_list, open(verif_sample_list_file, 'wb'))


def create_sample_list_batch_shuffle(root_folder, train_valid_sample_list_file):
    person_list = os.listdir(root_folder)
    train_sample_list = []
    valid_sample_list = []
    # 一个人平均有29张图片, 最多产生565.5对正样本, 所以没565个人进行一次shuffle, 在训练时, 一次读入16385个图片
    # 用于训练FaceNet等pair类型的模型
    batch_train_valid_sample_list = []
    for person_index, person_name in enumerate(person_list[:-2000]):
        person_path = os.path.join(root_folder, person_name)
        pic_list = map(lambda y: (y, person_index), map(lambda x: os.path.join(person_path, x), os.listdir(person_path)))
        if person_index > 0 and person_index % 565 == 0:
            np.random.shuffle(batch_train_valid_sample_list)
            train_num = int(len(batch_train_valid_sample_list) * 0.8)
            batch_train_sample_list = batch_train_valid_sample_list[:train_num]
            batch_valid_sample_list = batch_train_valid_sample_list[train_num:]
            train_sample_list.extend(batch_train_sample_list)
            valid_sample_list.extend(batch_valid_sample_list)
            batch_train_valid_sample_list = []
        else:
            batch_train_valid_sample_list.extend(pic_list)
    np.random.shuffle(batch_train_valid_sample_list)
    train_num = int(len(batch_train_valid_sample_list) * 0.8)
    batch_train_sample_list = batch_train_valid_sample_list[:train_num]
    batch_valid_sample_list = batch_train_valid_sample_list[train_num:]
    train_sample_list.extend(batch_train_sample_list)
    valid_sample_list.extend(batch_valid_sample_list)

    msgpack_numpy.dump((train_sample_list, valid_sample_list), open(train_valid_sample_list_file, 'wb'))


if __name__ == '__main__':
    root_folder = '/data/liubo/face/MS-Celeb_face_filter'
    train_valid_sample_list_file = '/data/liubo/face/MS-Celeb_face_list/sample_list_batch_shuffle.p'
    verif_sample_list_file =  '/data/liubo/face/MS-Celeb_face_list/sample_list_verif.p'
    # create_sample_list(root_folder, train_valid_sample_list_file, verif_sample_list_file)
    create_sample_list_batch_shuffle(root_folder, train_valid_sample_list_file)
