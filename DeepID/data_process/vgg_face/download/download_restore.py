#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: download_.py
@time: 2016/5/20 14:18
"""

import base64
from time import time
import os


def restore(pic_str, pic_file_name):
    if 'error' in pic_str:
        return
    f = open(pic_file_name, 'w')
    pic = base64.b64decode(pic_str)
    f.write(pic)
    f.close()


def main_restore():
    hadoop_data_folder = '/data/liubo/face/vgg_face_dataset/all_data/hadoop_data'
    all_pic_folder = '/data/liubo/face/vgg_face_dataset/all_data/pictures'
    file_list = os.listdir(hadoop_data_folder)
    for file_name in file_list:
        absolute_path = os.path.join(hadoop_data_folder, file_name)
        count = 0
        start = time()
        name = ''
        for line in open(absolute_path):
            tmp = line.rstrip().split()
            try:
                if len(tmp) >= 3:
                    name = tmp[0]
                    pic_folder = os.path.join(all_pic_folder, name)
                    if not os.path.exists(pic_folder):
                        os.makedirs(pic_folder)
                    pic_file_name = os.path.join(pic_folder, tmp[1]+'.png')
                    pic_str = tmp[2]
                    restore(pic_str, pic_file_name)
                    count += 1
            except:
                continue
                # print line
                # continue
        end = time()
        print name, count, (end-start)


if __name__ == '__main__':
    main_restore()