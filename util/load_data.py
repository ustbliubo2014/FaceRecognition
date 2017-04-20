# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: load_data.py
@time: 2016/7/4 15:19
@contact: ustb_liubo@qq.com
@annotation: load_data : 负责读入不同尺寸,不同维度的数据; 包含数据的预处理(旋转图像,切分图像[分割patch])
[最后每个函数都是通过函数名+args调用]
MyThread(func=writer, args=valid_write_args, name='valid_write')
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import traceback
from scipy.misc import imread, imresize, imsave
import numpy as np
from keras.utils import np_utils
from PIL import Image
import os
from skimage.transform import rotate
from random import randint
import pdb


def rotate_img(img, angle):
    # 旋转图片
    return rotate(img, randint(-angle, angle))


def flip_lr(img):
    # 水平翻转图片
    if randint(0, 1):
        return np.fliplr(img)
    else:
        return img


def flip_ud(img):
    # 垂直翻转图片
    return np.flipud(img)


def read_one_rgb_pic(pic_path, pic_shape, func_args_dic):
    '''
    :param pic_path: 文件路径
    :param pic_shape: 最后图片的shape
    :param func_dic: 对该图片的处理 {func:args(该函数的参数[旋转图片时需要旋转角度])} [一张图片可能需要多个]
    :return:
    '''
    # 读入一张rgb图片
    if len(func_args_dic) == 0:
        return imresize(imread(pic_path), pic_shape)
    img = imread(pic_path)
    # 先读入图片,在做各种处理,最后修改尺寸
    print func_args_dic
    for func in func_args_dic:
        args = [img]
        args.extend(func_args_dic.get(func))
        img = func(*args)
    return imresize(img, pic_shape)


def read_one_gray_pic(pic_path, pic_shape, func_args_dic):
    '''
    :param pic_path: 文件路径
    :param pic_shape: 最后图片的shape
    :param func_dic: 对该图片的处理 {func:args(该函数的参数[旋转图片时需要旋转角度])} [一张图片可能需要多个]
    :return:
    '''
    # 读入一张rgb图片
    if len(func_args_dic) == 0:
        return imresize(np.array(Image.open(pic_path).convert('L')), pic_shape)
    img = np.array(Image.open(pic_path).convert('L'))
    # 先读入图片,在做各种处理,最后修改尺寸
    print func_args_dic
    for func in func_args_dic:
        args = [img]
        args.extend(func_args_dic.get(func))
        img = func(*args)
    return imresize(img, pic_shape)



def load_rgb_batch_data(batch_sample_list, person_num, pic_shape, func_args_dic):
    '''
        :param sample_list: [(path_1,label_1),...,(path_n, label_n)]
        :param person_num: label维度,用于生成数据
        :param pic_shape:
        :param 对图片的操作(取某一部分/旋转/平移)
        :return: 每次读入部分数据到队列中, 边训练边读入
    '''
    X = []
    Y = []
    for sample_path,person_id in batch_sample_list:
        try:
            X.append(read_one_rgb_pic(sample_path, pic_shape, func_args_dic))
            Y.append(person_id)
        except:
            traceback.print_exc()
            continue
    X = np.asarray(X, dtype=np.float32) / 255.0
    X = np.transpose(X,(0,3,1,2))
    Y = np_utils.to_categorical(np.asarray(Y, dtype=np.int), person_num)
    return X, Y


def load_gray_batch_data(batch_sample_list, person_num, pic_shape, func_args_dic):
    '''
        :param sample_list: [(path_1,label_1),...,(path_n, label_n)]
        :param person_num: label维度,用于生成数据
        :param pic_shape:
        :param 对图片的操作(取某一部分/旋转/平移)
        :return: 每次读入部分数据到队列中, 边训练边读入
    '''
    X = []
    Y = []
    for sample_path, person_id in batch_sample_list:
        X.append(read_one_gray_pic(sample_path, pic_shape, func_args_dic))
        Y.append(person_id)
    X = np.asarray(X, dtype=np.float32) / 255.0
    X = np.reshape(X, (X.shape[0], 1, X.shape[1], X.shape[2]))
    Y = np_utils.to_categorical(np.asarray(Y, dtype=np.int), person_num)
    return X, Y


def load_rgb_multi_person_all_data(all_person_folder, pic_shape, label_int, person_num_threshold,pic_num_threshold,
                                   filter_list, func_args_dic):
    '''
        :param all_person_folder: 多个文件夹,每个文件夹是一个人
        :param pic_shape:
        :param label_int: True则返回的label是float, False返回label是string
        :param person_num_threshold:需要读入多少人的图片(可以读入少量人的图片进行测试)
        :param pic_num_threshold: 每个人读取多少张图片(每个人读入少量图片进行测试)
        :param filter_list: 包含哪些字段不用读入['unknown', 'Must_Same', 'Maybe_same']
        :param 对图片的操作(取某一部分/旋转/平移)
        :return:
    '''
    person_list = os.listdir(all_person_folder)
    if person_num_threshold != None:
        person_num_threshold = min(person_num_threshold, len(person_list))
    data = []
    label = []
    all_pic_list = []
    for person_index, person in enumerate(person_list[:person_num_threshold]):
        if person_index % 10 == 0:
            print person_index
        if len(filter_list) > 0:
            has_filter = False
            for filter in filter_list:
                if person in filter:
                    has_filter = True
                    break
            if has_filter:
                continue
        person_path = os.path.join(all_person_folder, person)
        if not os.path.isdir(person_path):
            print 'error dir :', person_path
            continue
        pic_list = os.listdir(person_path)
        if pic_num_threshold != None:
            this_pic_num_threshold = min(len(pic_list), pic_num_threshold)
        else:
            this_pic_num_threshold = len(pic_list)
        pic_list = pic_list[:this_pic_num_threshold]
        all_pic_list.extend(pic_list)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            try:
                data.append(read_one_rgb_pic(pic_path, pic_shape, func_args_dic))
                if label_int:
                    label.append(person_index)
                else:
                    label.append(person)
            except:
                traceback.print_exc()
                continue
    data = np.asarray(data, dtype=np.float32)
    data = data / 255.0
    data = np.transpose(data,(0,3,1,2))
    label = np.asarray(label)
    return data, label, all_pic_list


def load_gray_multi_person_all_data(all_person_folder, pic_shape, label_int, person_num_threshold,pic_num_threshold,
                                    filter_list, func_args_dic):
    '''
        :param all_person_folder: 多个文件夹,每个文件夹是一个人
        :param pic_shape:
        :param label_int: True则返回的label是float, False返回label是string
        :param person_num_threshold:需要读入多少人的图片(可以读入少量人的图片进行测试)
        :param pic_num_threshold: 每个人读取多少张图片(每个人读入少量图片进行测试)
        :param filter_list: 包含哪些字段不用读入['unknown', 'Must_Same', 'Maybe_same']
        :param 对图片的操作(取某一部分/旋转/平移)
        :return:
    '''
    person_list = os.listdir(all_person_folder)
    if person_num_threshold != None:
        person_num_threshold = min(person_num_threshold, len(person_list))
    data = []
    label = []
    all_pic_list = []
    for person_index, person in enumerate(person_list[:person_num_threshold]):
        if len(filter_list) > 0:
            has_filter = False
            for filter in filter_list:
                if person in filter:
                    has_filter = True
                    break
            if has_filter:
                continue
        person_path = os.path.join(all_person_folder, person)
        if not os.path.isdir(person_path):
            continue

        pic_list = os.listdir(person_path)
        if pic_num_threshold != None:
            pic_num_threshold = min(len(pic_list), pic_num_threshold)
        else:
            pic_num_threshold = len(pic_list)
        pic_list = pic_list[:pic_num_threshold]
        all_pic_list.extend(pic_list)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            try:
                data.append(read_one_gray_pic(pic_path, pic_shape, func_args_dic))
                if label_int:
                    label.append(person_index)
                else:
                    label.append(person)
            except:
                traceback.print_exc()
                continue
    data = np.asarray(data, dtype=np.float32)
    data = data / 255.0
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    label = np.asarray(label)
    return data, label, all_pic_list



def load_rgb_unknown_person_all_data(person_path, pic_shape, pic_num_threshold, func_args_dic):
    '''
        :param all_person_folder: 多个文件夹,每个文件夹是一个人
        :param pic_shape:
        :param pic_num_threshold: 需要读入多少图片
        :param 对图片的操作(取某一部分/旋转/平移)
        :return:
    '''
    data = []
    all_pic_list = []
    pic_list = os.listdir(person_path)
    if pic_num_threshold != None:
        pic_num_threshold = min(len(pic_list), pic_num_threshold)
    else:
        pic_num_threshold = len(pic_list)
    pic_list = pic_list[:pic_num_threshold]
    all_pic_list.extend(pic_list)
    for pic in pic_list:
        pic_path = os.path.join(person_path, pic)
        try:
            data.append(read_one_rgb_pic(pic_path, pic_shape, func_args_dic))
        except:
            traceback.print_exc()
            continue
    data = np.asarray(data, dtype=np.float32)
    data = data / 255.0
    data = np.transpose(data,(0,3,1,2))
    return data


def load_gray_unknown_person_all_data(person_path, pic_shape, pic_num_threshold, func_args_dic):
    '''
        :param all_person_folder: 多个文件夹,每个文件夹是一个人
        :param pic_shape:
        :param pic_num_threshold: 需要读入多少图片
        :param 对图片的操作(取某一部分/旋转/平移)
        :return:
    '''
    data = []
    all_pic_list = []
    pic_list = os.listdir(person_path)
    if pic_num_threshold != None:
        pic_num_threshold = min(len(pic_list), pic_num_threshold)
    else:
        pic_num_threshold = len(pic_list)
    pic_list = pic_list[:pic_num_threshold]
    all_pic_list.extend(pic_list)
    for pic in pic_list:
        pic_path = os.path.join(person_path, pic)
        try:
            data.append(read_one_gray_pic(pic_path, pic_shape, func_args_dic))
        except:
            traceback.print_exc()
            continue
    data = np.asarray(data, dtype=np.float32)
    data = data / 255.0
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    return data


def load_two_deep_path(folder):
    '''
        读入两级目录的文件列表 : 每个人最多取50张图片
        annotate
            liubo
                person_1
                person_2
            ligang
                person_3
                person_4
        :param folder: 文件名
        :return: {person:[pic_path1,pic_path2]}
    '''
    folder_list = os.listdir(folder)
    person_path_dic = {}

    for child_folder in folder_list:
        child_folder_path = os.path.join(folder, child_folder)
        person_list = os.listdir(child_folder_path)
        for person in person_list:
            person_path = os.path.join(child_folder_path, person)
            path_list = map(lambda x:os.path.join(person_path, x), os.listdir(person_path))
            person_path_dic[person] = path_list
    return person_path_dic


def load_one_deep_path(folder):
    '''
        读入两级目录的文件列表 : 每个人最多取50张图片
        vgg_all
            person_1
            person_2
            person_3
            person_4
        :param folder: 文件名
        :return: {person:[pic_path1,pic_path2]}
    '''
    person_list = os.listdir(folder)
    person_path_dic = {}
    for person in person_list:
        person_path = os.path.join(folder, person)
        path_list = map(lambda x:os.path.join(person_path, x), os.listdir(person_path))
        person_path_dic[person] = path_list
    return person_path_dic


def split_data(data, label, split_rate=0.9):
    np.random.seed(1337)
    shuffle_list = range(data.shape[0])
    np.random.shuffle(shuffle_list)
    train_num = int(data.shape[0] * split_rate)
    train_data = data[shuffle_list[:train_num]]
    train_label = label[shuffle_list[:train_num]]
    valid_data = data[shuffle_list[train_num:]]
    valid_label = label[shuffle_list[train_num:]]
    return train_data, train_label, valid_data, valid_label


def load_pair_data(person_path_dic, pre_person_num=10):
    '''
        将数据以pair的形式读入
        :param person_path_dic: 人名, path 列表
        :param pre_person_num: 每个人的正样本数(正负样本数相同)
        :return:
    '''
    person_list = person_path_dic.keys()
    person_num = len(person_list) - 1
    has_add_pair_dic = {}
    for person in person_path_dic:
        pic_path_list = person_path_dic.get(person)
        # 得到正样本
        if len(pic_path_list) < 10:
            continue
        length = len(pic_path_list) - 1
        count = 0
        while count < pre_person_num:
            first = randint(0, length)
            second = randint(0, length)
            if first == second:
                continue
            if first > second:
                first, second = second, first
            pair = pic_path_list[first] + '\t' + pic_path_list[second]
            if pair in has_add_pair_dic:
                continue
            else:
                has_add_pair_dic[pair] = 0
                count += 1
        # 得到负样本
        count = 0
        while count < pre_person_num:
            other_person = person_list[randint(0, person_num)]
            if other_person == person:
                continue
            pair = pic_path_list[randint(0, length)] + '\t' + person_path_dic.get(other_person)[0]
            if pair in has_add_pair_dic:
                continue
            else:
                has_add_pair_dic[pair] = 1
                count += 1
    return has_add_pair_dic


if __name__ == '__main__':

    # pic_path = 'liubo3_flip.png'
    # func_args_dic = {flip_lr: (), rotate_img: (30,)}
    # imsave('new.png', read_one_gray_pic(pic_path, pic_shape=(128,128), func_args_dic=func_args_dic))
    folder = '/data/pictures_annotate/'
    pair_file = '/data/annotate_pair.txt'

    person_path_dic = load_two_deep_path(folder)
    has_add_pair_dic = load_pair_data(person_path_dic)
    f = open(pair_file, 'w')
    for pair in has_add_pair_dic:
        f.write(pair+'\t'+str(has_add_pair_dic.get(pair))+'\n')
    f.close()
    pass