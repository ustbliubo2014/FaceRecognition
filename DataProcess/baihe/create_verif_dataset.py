# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: create_verif_dataset.py
@time: 2017/1/5 19:25
@contact: ustb_liubo@qq.com
@annotation: create_verif_dataset : 构造一个类似于lfw的数据集(5000个人, 每个人两个正样本, 两个负样本)
"""
import sys
import logging
from logging.config import fileConfig
import os
import shutil
import numpy as np
import random
import msgpack_numpy
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def split_data(raw_folder, dst_folder):
    pic_num = 0
    small_person_num = 0
    no_person_num = 0
    other_person_num = 0
    count = 0
    for root_folder, sub_folder_list, sub_file_list in os.walk(raw_folder):
        count += 1
        person_name = os.path.split(root_folder)[1]
        this_pic_num = len(sub_file_list)
        dst_person_folder = os.path.join(dst_folder, person_name)
        if this_pic_num == 0 and len(sub_folder_list) == 0:
            print 'del :', root_folder
            # shutil.rmtree(root_folder)
            no_person_num += 1
        elif this_pic_num > 0 and this_pic_num < 5 and len(sub_folder_list) == 0:
            print 'move :', dst_person_folder
            small_person_num += 1
            pic_num += this_pic_num
            shutil.move(root_folder, dst_person_folder)
        else:
            print 'stay :'
            other_person_num += 1
        print count
    print 'small_person_num :', small_person_num, 'no_person_num :', no_person_num, 'other_person_num :', other_person_num
    # small_person_num: 5505 no_person_num: 184 other_person_num: 34125


def create_lfw_pair(folder, pair_file):
    person_list = os.listdir(folder)
    tmp_list = []
    # 选择正样本
    for person in person_list:
        person_path = os.path.join(folder, person)
        pic_list = map(lambda x:os.path.join(os.path.join(person_path, x)), os.listdir(person_path))
        if len(pic_list) > 2:
            # 每个人选择一个正样本
            np.random.shuffle(pic_list)
            tmp_list.append((pic_list[0], pic_list[1], True))
    # 选择相同数量的负样本
    np.random.shuffle(person_list)
    person_num = len(person_list)
    positive_num = len(tmp_list)
    count = 0
    for person_index, person in enumerate(person_list):
        this_person_path = os.path.join(folder, person)
        pic_list = map(lambda x:os.path.join(os.path.join(this_person_path, x)), os.listdir(this_person_path))
        other_person = person_list[(person_index+1)%person_num]
        other_person_path = os.path.join(folder, other_person)
        other_pic_list = map(lambda x:os.path.join(os.path.join(other_person_path, x)), os.listdir(other_person_path))
        np.random.shuffle(pic_list)
        np.random.shuffle(other_pic_list)
        tmp_list.append((pic_list[0], other_pic_list[0], False))
        count += 1
        if count == positive_num:
            break
    pair_list = []
    label_list = []
    np.random.shuffle(tmp_list)
    num = len(tmp_list)
    tmp_list = tmp_list[:num/100*100]
    for element in tmp_list:
        pic_path1, pic_path2, label = element
        pair_list.append(pic_path1)
        pair_list.append(pic_path2)
        label_list.append(label)
    print len(tmp_list), len(pair_list), len(label_list)
    msgpack_numpy.dump((pair_list, label_list), open(pair_file, 'wb'))


def create_lfw_pair_txt(folder, pair_file):
    person_list = os.listdir(folder)
    tmp_list = []
    # 选择正样本
    for person in person_list:
        person_path = os.path.join(folder, person)
        pic_list = map(lambda x:os.path.join(os.path.join(person_path, x)), os.listdir(person_path))
        if len(pic_list) > 2:
            # 每个人选择一个正样本
            np.random.shuffle(pic_list)
            tmp_list.append((pic_list[0], pic_list[1], True))
    # 选择相同数量的负样本
    np.random.shuffle(person_list)
    person_num = len(person_list)
    positive_num = len(tmp_list)
    count = 0
    for person_index, person in enumerate(person_list):
        this_person_path = os.path.join(folder, person)
        pic_list = map(lambda x:os.path.join(os.path.join(this_person_path, x)), os.listdir(this_person_path))
        other_person = person_list[(person_index+1)%person_num]
        other_person_path = os.path.join(folder, other_person)
        other_pic_list = map(lambda x:os.path.join(os.path.join(other_person_path, x)), os.listdir(other_person_path))
        np.random.shuffle(pic_list)
        np.random.shuffle(other_pic_list)
        tmp_list.append((pic_list[0], other_pic_list[0], False))
        count += 1
        if count == positive_num:
            break
    np.random.shuffle(tmp_list)
    f_pair = open(pair_file, 'w')
    f_pair.write('10'+'\t'+'300'+'\n')
    true_list = []
    false_list = []
    for element in tmp_list:
        pic_path1, pic_path2, label = element
        if label == True:
            true_list.append(element)
        else:
            false_list.append(element)
    num = min(len(true_list), 3000)
    true_list = true_list[:num/100*100]
    false_list = false_list[:num/100*100]
    num = len(true_list) / 10
    result_list = []
    for index in range(10):
        result_list.extend(true_list[index*num:(index+1)*num])
        result_list.extend(false_list[index*num:(index+1)*num])
    for element in result_list:
        pic_path1, pic_path2, label = element
        tmp1 = pic_path1.split('/')
        tmp2 = pic_path2.split('/')
        if label == True:
            f_pair.write('\t'.join([tmp1[-2], str(int(tmp1[-1][-8:-4])), str(int(tmp2[-1][-8:-4]))])+'\n')
        else:
            f_pair.write('\t'.join([tmp1[-2], str(int(tmp1[-1][-8:-4])), tmp2[-2], str(int(tmp2[-1][-8:-4]))]) + '\n')


if __name__ == '__main__':
    # split_data(raw_folder='/data/liubo/face/baihe/person_dlib_face',
    #            dst_folder='/data/liubo/face/baihe/person_dlib_face_verif')
    # create_lfw_pair(folder='/data/liubo/face/baihe/person_dlib_face_verif',
    #                 pair_file='/data/liubo/face/baihe/person_dlib_face_verif_pair.txt')
    #
    # split_data(raw_folder='/data/liubo/face/baihe/person_mtcnn_96',
    #            dst_folder='/data/liubo/face/baihe/person_mtcnn_96_verif')
    # create_lfw_pair_txt(folder='/data/liubo/face/baihe/verif/person_dlib_face_5pic',
    #                  pair_file='/data/liubo/face/baihe/verif/person_dlib_face_5pic_pair.txt')
    create_lfw_pair_txt(folder='/data/liubo/face/baihe/verif/person_mtcnn_160_5pic',
                        pair_file='/data/liubo/face/baihe/verif/person_mtcnn_160_5pic_pair.txt')
    pass
