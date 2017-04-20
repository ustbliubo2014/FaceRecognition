# -*-coding:utf-8 -*-
__author__ = 'liubo-it'

import os
from DeepID.DeepId1.DeepId import train_valid_deepid, train_deepid_batch, build_deepid_model, \
    valid_deepid_batch, create_deepId_network
import pdb
import numpy as np
from time import time
from keras.utils import np_utils
import msgpack
from scipy.misc import imread, imresize

pic_shape = (128, 128, 3)
input_shape = (3, 128, 128)
nb_classes = 2622
weight_file = '/data/liubo/face/vgg_face_dataset/all_data/model/vgg_face.all_data.big.rgb.bn.deepid.weight'
model_file = '/data/liubo/face/vgg_face_dataset/all_data/model/vgg_face.all_data.big.rgb.bn.deepid.model'

'''
    效果不好,不在使用
'''

def load_data(small_train_list):
    # [(path,label)]
    all_data = []
    all_label = []
    for path,label in small_train_list:
        try:
            all_data.append(imresize(imread(path), pic_shape))
            all_label.append(int(label))
        except:
            print 'error file :', path
            continue
    all_data = np.asarray(all_data, dtype=np.float32)
    all_label = np.asarray(all_label, dtype=np.float32)
    return all_data, all_label


def train_all_vgg_face():

    sample_list_file = '/data/liubo/face/vgg_face_dataset/all_data/all_sample_list.p'
    train_sample_list, valid_sample_list = msgpack.load(open(sample_list_file,'rb'))
    train_small_data_num = 10000 # 每次读取10万条数据
    valid_small_data_num = 2000
    epoch_num = 5
    model = build_deepid_model(create_deepId_network, input_shape, nb_classes)
    open(model_file,'w').write(model.to_json())

    if os.path.exists(weight_file):
        print 'load_weights'
        model.load_weights(weight_file)

    last_mean_acc = 0
    for index in range(epoch_num):
        np.random.shuffle(train_sample_list)
        size = len(train_sample_list) / train_small_data_num
        for k in range(size):
            start = time()
            small_train_list = train_sample_list[k*train_small_data_num:(k+1)*train_small_data_num]
            all_data, all_label = load_data(small_train_list)
            all_data = np.transpose(all_data, (0,3,1,2))
            all_label = np_utils.to_categorical(all_label, nb_classes)
            print all_data.shape, all_label.shape, (time()-start)
            train_deepid_batch(all_data, all_label, nb_classes, model, weight_file)
        if train_small_data_num*size < len(train_sample_list):
            small_train_list = train_sample_list[train_small_data_num*size:]
            all_data, all_label = load_data(small_train_list)
            all_data = np.transpose(all_data, (0,3,1,2))
            all_label = np_utils.to_categorical(all_label, nb_classes)
            print all_data.shape, all_label.shape, (time()-start)
            train_deepid_batch(all_data, all_label, nb_classes, model, weight_file)

        this_epoch_acc = []
        np.random.shuffle(valid_sample_list)
        size = len(valid_sample_list) / valid_small_data_num
        for k in range(size):
            start = time()
            small_valid_list = train_sample_list[k*valid_small_data_num:(k+1)*valid_small_data_num]
            all_data, all_label = load_data(small_valid_list)
            all_data = np.transpose(all_data, (0,3,1,2))
            all_label = np_utils.to_categorical(all_label, nb_classes)
            print all_data.shape, all_label.shape, (time()-start)
            this_epoch_acc.append(valid_deepid_batch(all_data, all_label, model, weight_file))
        if valid_small_data_num*size < len(valid_sample_list):
            small_valid_list = valid_sample_list[valid_small_data_num*size:]
            all_data, all_label = load_data(small_valid_list)
            all_data = np.transpose(all_data, (0,3,1,2))
            all_label = np_utils.to_categorical(all_label, nb_classes)
            print all_data.shape, all_label.shape, (time()-start)
            this_epoch_acc.append(valid_deepid_batch(all_data, all_label, model, weight_file))

        this_mean_acc = np.mean(this_epoch_acc)
        print 'this_mean_acc :', this_mean_acc
        if this_mean_acc > last_mean_acc:
            print 'save model'
            model.save_weights(weight_file,overwrite=True)
            last_mean_acc = this_epoch_acc


if __name__ == '__main__':
    train_all_vgg_face()


