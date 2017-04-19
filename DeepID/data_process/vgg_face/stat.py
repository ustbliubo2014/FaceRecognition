# -*- coding:utf-8 -*-
__author__ = 'liubo-it'

import os

if __name__ == '__main__':
    father_folder = '/data/liubo/face/vgg_face_dataset/pictures'
    folder_list = os.listdir(father_folder)
    for folder in folder_list:
        print folder, len(os.listdir(os.path.join(father_folder, folder)))