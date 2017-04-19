# -*-coding:utf-8 -*-
__author__ = 'liubo-it'

# 将所有图片转换成指定大小

from scipy.misc import imread, imsave, imresize
import os
import shutil
from time import time

def trans_all_pic(vgg_pic_align_folder, new_pic_size=(128, 128, 3)):
    person_list = os.listdir(vgg_pic_align_folder)
    for person in person_list:
        start = time()
        absolute_folder_path = os.path.join(vgg_pic_align_folder, person)
        pic_list = os.listdir(absolute_folder_path)
        for pic in pic_list:
            absolute_pic_path = os.path.join(absolute_folder_path, pic)
            try:
                imsave(absolute_pic_path, imresize(imread(absolute_pic_path), new_pic_size))
            except:
                print 'rm file : ', absolute_pic_path
                os.remove(absolute_pic_path)
                continue
        end = time()
        print person, (end-start)

if __name__ == '__main__':
    vgg_pic_align_folder = '/data/liubo/face/vgg_face_dataset/picture_align'
    trans_all_pic(vgg_pic_align_folder)
