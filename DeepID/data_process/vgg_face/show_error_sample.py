#-*- coding:utf-8 -*-
__author__ = 'liubo-it'

# 将分类错误的图片显示出来

from scipy.misc import imsave
import os
import msgpack_numpy
import shutil
import pdb

def show_error_sample(X_data, error_sample_file, label_trans_dic, error_sample_folder):
    if os.path.exists(error_sample_folder):
        shutil.rmtree(error_sample_folder)
    os.makedirs(error_sample_folder)
    for line in open(error_sample_file):
        if line.startswith('index'):
            continue
        tmp = line.rstrip().split('\t')
        index = int(tmp[0])
        y_true = int(tmp[1])
        y_predict = int(tmp[2])
        true_person = label_trans_dic.get(y_true)
        predict_person = label_trans_dic.get(y_predict)
        arr = X_data[index]
        pic_name = os.path.join(error_sample_folder, str(index)+'---'+true_person+'---'+predict_person+'.png')
        imsave(pic_name, arr)

def find_error_sample_index(error_sample_folder, del_error_sample_folder):
    error_sample_index_list = []
    error_list = set(os.listdir(error_sample_folder))
    del_error_list = set(os.listdir(del_error_sample_folder))
    del_list = error_list - del_error_list
    for del_index in del_list:
        tmp = del_index.split('---')
        if len(tmp) == 3:
            error_sample_index_list.append(int(tmp[0]))
    return error_sample_index_list

def main_show_error_sample():
    pack_file = '/data/liubo/face/vgg_face_dataset/train_valid_data.p'
    error_train_sample_file = '/data/liubo/face/vgg_face_dataset/error_train_sample.txt'
    error_valid_sample_file = '/data/liubo/face/vgg_face_dataset/error_valid_sample.txt'
    error_train_sample_folder = '/data/liubo/face/vgg_face_dataset/error_train_sample'
    error_valid_sample_folder = '/data/liubo/face/vgg_face_dataset/error_valid_sample'
    train_data, train_label, valid_data, valid_label, label_trans_dic = msgpack_numpy.load(open(pack_file, 'rb'))
    show_error_sample(train_data, error_train_sample_file, label_trans_dic, error_train_sample_folder)
    show_error_sample(valid_data, error_valid_sample_file, label_trans_dic, error_valid_sample_folder)

if __name__ == '__main__':
    main_show_error_sample()
    pass