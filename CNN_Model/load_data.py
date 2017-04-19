# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: load_data.py
@time: 2016/8/8 19:02
@contact: ustb_liubo@qq.com
@annotation: load_data
"""
import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
from scipy.misc import imread, imsave, imresize
import pdb
import msgpack_numpy
from time import time
import ImageAugmenter
import traceback
import cv2
from sklearn.cross_validation import train_test_split
import keras.backend as K

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

avg = np.array([129.1863, 104.7624, 93.5940])
size = 96
pic_shape = (size, size, 3)

augmenter = ImageAugmenter.ImageAugmenter(size, size,
                                      hflip=True, vflip=False,
                                      scale_to_percent=1.5, scale_axis_equally=True,
                                      rotation_deg=0, shear_deg=0,
                                      translation_x_px=16, translation_y_px=16,
                                      channel_is_first_axis=False)


def read_one_rgb_pic(pic_path, pic_shape=pic_shape, need_augment=False):
    img = imread(pic_path)[:, :, :3]
    img = imresize(img, pic_shape)
    if need_augment:
        img = np.reshape(img, (1, pic_shape[0], pic_shape[1], pic_shape[2]))
        img = augmenter.augment_batch(img)[0] * 255
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def load_data_from_list(sample_list, pic_shape, need_augment=False):
    start = time()
    all_data = []
    all_label = []
    for pic_path, pic_label in sample_list:
        try:
            img = read_one_rgb_pic(pic_path, pic_shape, need_augment)[0]
            all_data.append(img)
            all_label.append(pic_label)
        except:
            traceback.print_exc()
            continue
    try:
        all_label = np.asarray(all_label)
        all_data = np.asarray(all_data)
        if K.image_dim_ordering() != 'th':
            all_data = np.transpose(all_data, (0, 2, 3, 1))
    except:
        traceback.print_exc()
        return None, None
    print all_data.shape, all_label.shape, (time() - start)
    return all_data, all_label


def create_sample_list(folder):
    sample_list = []
    current_person_index = 0
    person_list = os.listdir(folder)
    person_index = 1
    for person in person_list:
        print person_index, person
        person_path = os.path.join(folder, person)
        pic_list = map(lambda x: os.path.join(person_path, x), os.listdir(person_path))
        pic_list = map(lambda x:(x, person_index), pic_list)
        if len(pic_list) >= 15:
            sample_list.extend(pic_list)
            current_person_index += 1
            person_index += 1
    print 'all_pic_num :', len(sample_list),
    new_sample_list = []
    count = 0
    start = time()
    for path, person_index in sample_list[:]:
        try:
            # arr = imread(path)
            # if arr.shape[0] < 40 or arr.shape[1] < 40:
            #     print 'small pic :', path, arr.shape
            #     continue
            # else:
                new_sample_list.append((path, person_index))
                count += 1
                if count % 50000 == 0:
                    print count, time() - start
                    start = time()
        except:
            print 'error path :', path
            os.remove(path)
    return new_sample_list


def create_train_valid_list(folder):
    '''
        每个类别, 一个用于训练, 其余用于测试
    '''
    person_list = os.listdir(folder)
    train_list = []
    valid_list = []
    for person_index, person in enumerate(person_list):
        print person_index, person
        person_path = os.path.join(folder, person)
        pic_list = map(lambda x: os.path.join(person_path, x), os.listdir(person_path))
        pic_list = map(lambda x:(x, person_index), pic_list)
        if len(pic_list) <= 1:
            continue
        else:
            valid_list.append(pic_list[0])
            train_list.extend(pic_list[1:])
    print 'train_num :', len(train_list), 'valid_num :', len(valid_list)
    return train_list, valid_list


def stat(folder):
    pic_num_list = []
    current_person_index = 0
    all_num = 0
    for root_dir, dir_list, pic_list in os.walk(folder):
        if len(pic_list) >= 10:
            pic_num_list.append((root_dir, len(pic_list)))
            all_num += len(pic_list)
            print len(pic_num_list), all_num
            current_person_index += 1


if __name__ == '__main__':
    pass
    # stat(folder='/data/liubo/face/baihe/person_mtcnn_160')
    #
    #

    new_sample_list = create_sample_list('/data/liubo/face/baihe/person_dlib_face')
    train_sample_list, valid_sample_list = train_test_split(new_sample_list, test_size=0.1)
    print len(train_sample_list), len(valid_sample_list)
    msgpack_numpy.dump((train_sample_list, valid_sample_list),
                       open('/data/liubo/face/baihe/person_dlib_face_sample_list_30.p', 'wb'))


    # train_sample_list, valid_sample_list = create_train_valid_list('/data/liubo/face/baihe/baihe_person_face_align')
    # msgpack_numpy.dump((train_sample_list, valid_sample_list),
    #                    open('/data/liubo/face/baihe/face_align_train_valid_sample_list_filter.p', 'wb'))





