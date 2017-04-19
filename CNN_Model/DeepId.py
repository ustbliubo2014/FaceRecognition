# -*-coding:utf-8 -*-
__author__ = 'liubo-it'


import numpy as np
import pdb
np.random.seed(1337)  # for reproducibility
import random
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, merge, Dense, BatchNormalization, Dropout
from keras.layers.core import Flatten
from keras.utils import np_utils
import msgpack
from scipy.misc import imread, imresize
from time import time
import traceback
from keras import regularizers


def load_batch_data(batch_sample_list, person_num, pic_shape):
    X = []
    Y = []
    for sample_path,person_id in batch_sample_list:
        try:
            arr = imread(sample_path)
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                X.append(imresize(arr, pic_shape))
                Y.append(person_id)
        except:
            continue
    X = np.asarray(X, dtype=np.float32) / 255.0
    X = np.transpose(X,(0,3,1,2))
    Y = np_utils.to_categorical(np.asarray(Y, dtype=np.int), person_num)
    return X, Y


def writer(queue, epoch_num, batch_size, sample_list, batch_num, person_num, pic_shape):
    '''
        :param queue: 共享队列
        :param epoch_num:
        :param batch_size:
        :param sample_list:样本位置(给出位置,直接读取)
        :param batch_num: len(sample_list) / batch_size  --- 在train中也要知道读多少数据,需要给出epoch_num*batch_num
        :return:
    '''
    # person_num = len(set([tmp[1] for tmp in sample_list]))
    # 最后的结束有train_valid线程决定
    while True:
        np.random.shuffle(sample_list)
        for batch_id in range(batch_num):
            try:
                batch_x, batch_y = \
                    load_batch_data(sample_list[batch_id*batch_size:(batch_id+1)*batch_size], person_num, pic_shape)
                queue.put((batch_x, batch_y))
            except:
                continue


def euclidean_distance(vects):
    # 计算距离
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """
        对比损失,参考paper http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(all_img_arr, all_label):
    """
        生成对比样本, 同时还要保留原来的data和label,因为softmax需要data和label
        x : 读入的图片
        digit_indices : 图片对应的label label
    """
    SAME_PERSON = 1
    NO_SAME_PERSON = 0
    pairs_data = []
    pairs_label = []
    # 左右样本拆开
    first_data = []
    first_label = []
    second_data = []
    second_label = []
    # 转换数据格式 : {label : [img1, img2, ..., img_n]}
    label_img_list_dic = {}
    for index in range(len(all_label)):
        label = all_label[index]
        img_arr = all_img_arr[index]
        img_list = label_img_list_dic.get(label, [])
        img_list.append(img_arr)
        label_img_list_dic[label] = img_list
    label_list = label_img_list_dic.keys()
    label_num = len(label_list)

    for this_label in label_img_list_dic:
        # 产生所有可能的正样本
        this_positive_num =0
        img_list = label_img_list_dic.get(this_label)
        this_img_num = len(img_list)
        for index_i in range(this_img_num):
            for index_j in range(index_i + 1, this_img_num):
                first_data.append(img_list[index_i])
                first_label.append(this_label)
                second_data.append(img_list[index_j])
                second_label.append(this_label)
                pairs_data.append([img_list[index_i], img_list[index_j]])
                pairs_label.append(SAME_PERSON)
                this_positive_num += 1
        this_negative_num = 0
        while this_negative_num < this_positive_num:
            # 产生相同数量的负样本, 随机选人,然后选该人的第一张图片
            other_label = label_list[random.randint(0, label_num-1)]
            if other_label == this_label:
                continue
            this_img = img_list[random.randint(0, this_img_num-1)]
            other_img_list = label_img_list_dic.get(other_label)
            other_img = other_img_list[random.randint(0, len(other_img_list)-1)]
            first_data.append(this_img)
            first_label.append(this_label)
            second_data.append(other_img)
            second_label.append(this_label)
            pairs_data.append([this_img, other_img])
            pairs_label.append(NO_SAME_PERSON)
            this_negative_num += 1
    return np.array(pairs_data), np.array(pairs_label), np.asarray(first_data), np.asarray(first_label), \
                    np.asarray(second_data), np.asarray(second_label)


def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


def create_deepId_network(input_shape):
    print 'input_shape :', input_shape
    # (3, 128, 128)
    WEIGHT_DECAY = 0.0005

    input_layer = Input(shape=input_shape)
    layer0 = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(input_layer)
    # layer0 = BatchNormalization()(layer0)
    layer0 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer0)
    # layer0 = BatchNormalization()(layer0)
    layer0 = MaxPooling2D(pool_size=(2, 2))(layer0)


    layer1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer0)
    # layer1 = BatchNormalization()(layer1)
    layer1 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer1)
    # layer1 = BatchNormalization()(layer1)
    layer1 = MaxPooling2D(pool_size=(2, 2))(layer1)

    layer2 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer1)
    # layer2 = BatchNormalization()(layer2)
    layer2 = Convolution2D(nb_filter=256, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer2)
    # layer2 = BatchNormalization()(layer2)
    layer2 = MaxPooling2D(pool_size=(2, 2))(layer2)

    layer3 = Convolution2D(nb_filter=256, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer2)
    # layer3 = BatchNormalization()(layer3)
    layer3 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer3)
    # layer3 = BatchNormalization()(layer3)
    layer3 = MaxPooling2D(pool_size=(2, 2))(layer3)

    layer4 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer3)
    # layer4 = BatchNormalization()(layer4)
    layer4 = Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu',
                           border_mode='same', W_regularizer=regularizers.l2(WEIGHT_DECAY))(layer4)
    # layer4 = BatchNormalization()(layer4)

    layer3_flatten = Flatten()(layer3)
    layer4_flatten = Flatten()(layer4)
    flatten_layer = merge(inputs=[layer3_flatten, layer4_flatten], mode='concat')
    flatten_layer = Dropout(0.5)(flatten_layer)

    dense = Dense(512, activation='relu')(flatten_layer) # 在softmax前增加一个隐含层
    dense = Dropout(0.5)(dense)
    model = Model(input=[input_layer], output=[dense])

    print model.summary()
    
    return model


if __name__ == '__main__':
    input_shape = (3, 128, 128)
    create_deepId_network(input_shape)