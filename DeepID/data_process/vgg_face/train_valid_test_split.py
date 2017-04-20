# -*-coding:utf-8 -*-
__author__ = 'liubo-it'

# train_data 用于训练DeepId模型 ; valid_data 用于验证DeepId模型 ; test_data 用于提取特征,训练阈值 ;
# 先分train_valid和test,然后分train和valid ; 比例([100个person test,其余person train_valid]) ; train : valid =  8 : 2


import os
import numpy as np
import shutil
import msgpack_numpy
from scipy.misc import imread,imsave, imresize
import pdb
from show_error_sample import find_error_sample_index
import traceback

def split_test_data(vgg_pic_align_folder, train_valid_data_folder, test_data_folder, test_person_threshold=80,
                    person_num_threshold=50):
    '''
    :param vgg_pic_align_folder:原始文件夹
    :param train_valid_data_folder: 训练集和验证集
    :param test_data_folder: 测试集
    :param test_person_threshold: 测试集人数
    :param person_num_threshold: 一个文件夹下的图片数量最少50个
    :return:None
    '''
    person_list = os.listdir(vgg_pic_align_folder)
    np.random.shuffle(person_list)
    test_person_num = 0
    train_person_num = 0
    if not os.path.exists(train_valid_data_folder):
        os.makedirs(train_valid_data_folder)
    if not os.path.exists(test_data_folder):
        os.makedirs(test_data_folder)

    for person in person_list:
        if test_person_num <= test_person_threshold:
            src_folder = os.path.join(vgg_pic_align_folder, person)
            dst_folder = os.path.join(test_data_folder, person)
            if len(os.listdir(os.path.join(vgg_pic_align_folder, person))) >= 50:
                shutil.move(src=src_folder,dst=dst_folder)
                test_person_num += 1
            else:
                shutil.rmtree(src_folder)
        else:
            src_folder = os.path.join(vgg_pic_align_folder, person)
            dst_folder = os.path.join(train_valid_data_folder, person)
            if len(os.listdir(os.path.join(vgg_pic_align_folder, person))) >= 50:
                shutil.move(src=src_folder,dst=dst_folder)
                train_person_num += 1
            else:
                shutil.rmtree(src_folder)
    print 'train_person_num', train_person_num, 'test_person_num', test_person_num


def main_split_train_test():
    vgg_pic_align_folder = '/data/liubo/face/vgg_face_dataset/pictures_align'
    train_valid_data_folder = '/data/liubo/face/vgg_face_dataset/train_valid'
    test_data_folder = '/data/liubo/face/vgg_face_dataset/test'
    split_test_data(vgg_pic_align_folder, train_valid_data_folder, test_data_folder)


def split_train_valid(data, label, split_rate=0.8):
    split_list = range(len(label))
    np.random.shuffle(split_list)
    train_num = int(len(label) * split_rate)
    train_data = data[split_list[:train_num]]
    train_label = label[split_list[:train_num]]
    test_data = data[split_list[train_num:]]
    test_label = label[split_list[train_num:]]
    return train_data, train_label, test_data, test_label


def pack_data(person_folder, pack_file, is_test, person_num_threshold=100):
    person_list = os.listdir(person_folder)
    all_data = []
    all_label = []
    label_trans_dic = {}
    current_label = 0
    pic_shape = (50, 50, 3)
    for person in person_list[:500]:
        try:
            if person == 'unknown':
                continue
            label_trans_dic[current_label] = person
            pic_folder = os.path.join(person_folder, person)
            pic_list = os.listdir(pic_folder)
            pic_list.sort()
            this_person_num = 0

            for pic in pic_list:
                try:
                    absolute_path = os.path.join(pic_folder, pic)
                    pic_arr = imread(absolute_path)
                    if pic_arr.shape[2] != 3:
                        continue
                    pic_arr = imresize(pic_arr, pic_shape)
                    if pic_arr.shape != pic_shape:
                        continue
                    all_data.append(pic_arr)
                    all_label.append(current_label)
                    this_person_num += 1
                    if this_person_num > person_num_threshold:
                        # 每个样本最多取100张图片
                        break
                except:
                    traceback.print_exc()
                    print 'error person :', person
                    continue
            # print person, this_person_num
            current_label += 1
        except:
            traceback.print_exc()
            continue

    all_data = np.asarray(all_data, dtype=np.float32)
    all_label = np.asarray(all_label)
    all_data = all_data / 255.0
    if is_test:
        # msgpack_numpy.dump((all_data, all_label, label_trans_dic), open(pack_file, 'wb'))
        return all_data, all_label, label_trans_dic
    else:
        train_data, train_label, test_data, test_label = split_train_valid(all_data, all_label)
        # print train_data.shape, train_label.shape, test_data.shape, test_label.shape
        # msgpack_numpy.dump((train_data, train_label, test_data, test_label, label_trans_dic), open(pack_file, 'wb'))
        return train_data, train_label, test_data, test_label


def del_error_data(error_index_list, data):
    # 现将array转换成list后删除相关位置的元素
    data = list(data)
    for index in error_index_list:
        data.pop(index)
    data = np.asarray(data)
    return data


def main_pack_data():
    person_folder = '/data/liubo/face/vgg_face_dataset/test/'
    pack_file = '/data/liubo/face/vgg_face_dataset/test_data.p'
    pack_data(person_folder, pack_file, is_test=True)

    # person_folder = '/data/liubo/face/vgg_face_dataset/train_valid/'
    # pack_file = '/data/liubo/face/vgg_face_dataset/train_valid_data.p'
    # pack_data(person_folder, pack_file, is_test=False)


def main_del_error_valid_sample():
    pack_file = '/data/liubo/face/vgg_face_dataset/train_valid_data.p'
    train_data, train_label, valid_data, valid_label, label_trans_dic = msgpack_numpy.load(open(pack_file, 'rb'))
    error_train_sample_folder = '/data/liubo/face/vgg_face_dataset/error_train_sample'
    error_valid_sample_folder = '/data/liubo/face/vgg_face_dataset/error_valid_sample'
    del_error_train_sample_folder = '/data/liubo/face/vgg_face_dataset/del_error_train_sample'
    del_error_valid_sample_folder = '/data/liubo/face/vgg_face_dataset/del_error_valid_sample'
    del_error_train_sample_index_list = find_error_sample_index(error_train_sample_folder,del_error_train_sample_folder)
    del_error_train_sample_index_list.sort()
    del_error_train_sample_index_list = \
        map(lambda x:x[0]-x[1], zip(del_error_train_sample_index_list, range(len(del_error_train_sample_index_list))))
    print del_error_train_sample_index_list
    # train_data = del_error_data(del_error_train_sample_index_list, train_data)
    train_label = del_error_data(del_error_train_sample_index_list, train_label)
    # imsave('train_error.png',train_data[del_error_train_sample_index_list[-1]])
    del_error_valid_sample_index_list = find_error_sample_index(error_valid_sample_folder, del_error_valid_sample_folder)
    del_error_valid_sample_index_list.sort()
    print del_error_valid_sample_index_list
    del_error_valid_sample_index_list = \
        map(lambda x:x[0]-x[1], zip(del_error_valid_sample_index_list, range(len(del_error_valid_sample_index_list))))
    # valid_data = del_error_data(del_error_valid_sample_index_list, valid_data)
    valid_label = del_error_data(del_error_valid_sample_index_list, valid_label)
    print train_data.shape, train_label.shape, valid_data.shape, valid_label.shape
    msgpack_numpy.dump((train_data, train_label, valid_data, valid_label, label_trans_dic), open(pack_file, 'wb'))
    # imsave('valid_error.png',valid_data[del_error_valid_sample_index_list[-1]])


def split_test(test_folder, test_train_folder, test_valid_folder):
    person_list = os.listdir(test_folder)
    split_rate = 0.7
    for person in person_list:
        print person
        pic_folder = os.path.join(test_folder, person)
        pic_list = os.listdir(pic_folder)
        train_num = int(len(pic_list) * split_rate)
        train_folder = os.path.join(test_train_folder, person)
        valid_folder = os.path.join(test_valid_folder, person)
        os.makedirs(train_folder)
        os.makedirs(valid_folder)
        for pic in pic_list[:train_num]:
            shutil.copy(os.path.join(pic_folder, pic), os.path.join(train_folder, pic))
        for pic in pic_list[train_num:]:
            shutil.copy(os.path.join(pic_folder, pic), os.path.join(train_folder, pic))


if __name__ == '__main__':
    # main_split_train_test()
    main_pack_data()
    # main_del_error_valid_sample()
    # split_test(test_folder='/data/liubo/face/vgg_face_dataset/test',
    #            test_train_folder='/data/liubo/face/vgg_face_dataset/test_opencv_train',
    #            test_valid_folder='/data/liubo/face/vgg_face_dataset/test_opencv_valid'
    #     )