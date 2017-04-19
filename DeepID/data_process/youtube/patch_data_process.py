# -*- coding:utf-8 -*-
__author__ = 'liubo-it'

'''
    将脸部图片转换成patch去处理  --- 处理youtube的数据,因为该数据有身份对应关系
    由于一个patch需要训练一个模型,每个patch作为一个文件夹
'''

import os
from scipy.misc import imread, imsave, imresize
import pdb


def slice_one_patch(arr, row_size, col_size, row_patch_num, col_patch_num, patch_size, prefix_folder, suffix_folder,
                    patch_start_num, raw_pic_file_name):
    '''
    :param arr: 原始图像
    :param row_size: 该 patch图像的行
    :param col_size: 该 patch图像的列
    :param row_patch_num: 切分时, 原始图像一行切成几个
    :param col_patch_num: 切分时, 原始图像一列切成几个
    :param patch_size: 切分patch 需要resize的尺寸
    :param prefix_folder: 存放的文件夹
    :param suffix_folder: 存放的文件夹 (prefix_folder + patch_id + suffix_folder = 最后的文件夹)
    :param patch_start_num: 文件名, 也用于训练时; 训练时,一个patch训练一个
    :param raw_pic_file_name: 原始文件的文件名
    :return:
    '''
    row_move = (arr.shape[0] - row_size) / row_patch_num
    col_move = (arr.shape[1] - col_size) / col_patch_num
    for i in range(row_patch_num):
        for j in range(col_patch_num):
            patch_id = str(patch_start_num+i*col_patch_num+j)
            father_folder = os.path.join(prefix_folder, patch_id, suffix_folder)
            if not os.path.exists(father_folder):
                os.makedirs(father_folder)
            imsave(os.path.join(father_folder, raw_pic_file_name),
                   imresize(arr[i*row_move:i*row_move+row_size, j*col_move:j*col_move+col_size, :], patch_size))


def slice_one_pic(dir_path, pic_file_name, prefix_folder, suffix_folder):
    '''
        将一张人脸图片切分成60个patch, 原始图片的大小(55, 47, 3), 切分后的图片为31*39 , 一个原始图片切成10*6
        切分后的图片不能太小, 太小的话没有区分性
    :param dir_path : pic所在的文件夹
    :param data_file: 脸部图片的文件名
    :param slice后的父目录, 一个包含60个子目录, 每个子目录是一个patch的数据
    :return:
    '''
    pic_arr = imread(os.path.join(dir_path,pic_file_name))
    # 尺寸一 (20, 15, 3)
    slice_one_patch(pic_arr, row_size=20, col_size=15, row_patch_num=3, col_patch_num=4, patch_size=(20, 15, 3),
            prefix_folder=prefix_folder, suffix_folder=suffix_folder, patch_start_num=0, raw_pic_file_name=pic_file_name)
    # 尺寸二 (25, 20, 3)
    slice_one_patch(pic_arr, row_size=25, col_size=20, row_patch_num=3, col_patch_num=4, patch_size=(25, 20, 3),
            prefix_folder=prefix_folder, suffix_folder=suffix_folder, patch_start_num=12, raw_pic_file_name=pic_file_name)
    # 尺寸三 (30, 25, 3)
    slice_one_patch(pic_arr, row_size=30, col_size=25, row_patch_num=3, col_patch_num=4, patch_size=(30, 25, 3),
            prefix_folder=prefix_folder, suffix_folder=suffix_folder, patch_start_num=24, raw_pic_file_name=pic_file_name)
    # 尺寸四 (35, 30, 3)
    slice_one_patch(pic_arr, row_size=35, col_size=30, row_patch_num=3, col_patch_num=4, patch_size=(35, 30, 3),
            prefix_folder=prefix_folder, suffix_folder=suffix_folder, patch_start_num=36, raw_pic_file_name=pic_file_name)
    # 尺寸五 (40, 35, 3)
    slice_one_patch(pic_arr, row_size=40, col_size=35, row_patch_num=3, col_patch_num=4, patch_size=(40, 35, 3),
            prefix_folder=prefix_folder, suffix_folder=suffix_folder, patch_start_num=48, raw_pic_file_name=pic_file_name)


def slice_all_pic(raw_pic_folder, prefix_folder):
    for dir_path, dir_names, file_names in os.walk(raw_pic_folder):
        if len(dir_names) == 0:
            print 'dir_path', dir_path
            for file_name in file_names:
                slice_one_pic(dir_path=dir_path, pic_file_name=file_name, prefix_folder=prefix_folder,
                              suffix_folder=dir_path.replace(raw_pic_folder,'')[1:])


if __name__ == '__main__':
    slice_all_pic('/home/data/dataset/images/youtube/aligned_youtube', '/home/data/dataset/images/youtube/patch')
