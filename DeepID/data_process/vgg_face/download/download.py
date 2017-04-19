#-*- coding: utf-8 -*-
__author__ = 'liubo-it'

import urllib2
from time import time,sleep
import os
pic_num_threshold = 100

f_error = open('error.txt','a')

def download(url_file, pic_father_folder):
    name = os.path.split(url_file)[1].split('.')[0]
    sub_folder = os.path.join(pic_father_folder, name)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    print 'dir', name
    if len(os.listdir(sub_folder)) > pic_num_threshold:
        return
    count = 0
    start = time()
    for line in open(url_file):
        try:
            tmp = line.rstrip().split()
            if len(tmp) > 2:
                pic_id = tmp[0]
                url = tmp[1]
                file_name = os.path.join(sub_folder, pic_id+'.png')
                if os.path.exists(file_name):
                    continue
                a = urllib2.urlopen(url, timeout=10)
                f = open(file_name, "wb")
                f.write(a.read())
                end = time()
                if count % 20 == 0:
                    print url, file_name, (end-start)
                    sleep(5)
                    start = time()
                    if count > pic_num_threshold:
                        return
                count += 1
        except :
            f_error.write('error'+line.rstrip()+'\n')
            continue


if __name__ == '__main__':
    # url_file_folder = '/data_1/vgg_face_dataset/files'
    # pic_father_folder = '/data_1/vgg_face_dataset/pictures'
    url_file_folder = 'D:\FaceRecognization//vgg_face_dataset/files'
    pic_father_folder = 'D:\FaceRecognization//vgg_face_dataset/pictures'
    url_file_list = os.listdir(url_file_folder)
    for url_file in url_file_list:
        download(os.path.join(url_file_folder,url_file), pic_father_folder)

