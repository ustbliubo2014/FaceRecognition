# -*-coding:utf-8 -*-
__author__ = 'liubo-it'

import os
from scipy.misc import imread, imresize, imsave
from time import time
import traceback
import pdb
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def load_url_file(url_file):
    dic = {} #{id : [top , left , bottem , right , pose , detection_score , curation]}
    for line in open(url_file):
        try:
            tmp = line.rstrip().split()
            if len(tmp) < 4:
                continue
            dic[tmp[1]] = tmp[3:]
        except:
            continue
    return dic

def get_bound_box(pic_folder, url_file, dest_folder, new_size=(128, 128)):
    start = time()
    dic = load_url_file(url_file)
    pic_list = os.listdir(pic_folder)
    count = 0
    for pic_name in pic_list:
        try:
            # pdb.set_trace()
            id = pic_name[:-4]
            attr = dic.get(id)
            left = int(float(attr[0]))
            top = int(float(attr[1]))
            right = int(float(attr[2]))
            bottom = int(float(attr[3]))
            absolute_path = os.path.join(pic_folder, pic_name)
            pic_arr = imread(absolute_path)
            bound_box = imresize(pic_arr[top:bottom, left:right, :], new_size)
            dest_absolute_path = os.path.join(dest_folder, pic_name)
            imsave(dest_absolute_path, bound_box)
            count += 1
        except:
            # traceback.print_exc()
            # print 'error file', absolute_path
            continue
    end = time()
    print pic_folder, (end - start), len(pic_list), count

def main():
    father_pic_folder = 'D:\data/face\FaceRecognization/vgg_face_dataset\pictures/'
    father_file_folder = 'D:\data/face\FaceRecognization/vgg_face_dataset/files/'
    father_pic_box_folder = 'D:\data/face\FaceRecognization/vgg_face_dataset\pictures_box'
    person_list = os.listdir(father_pic_folder)
    for person in person_list:
        try:
            print person
            pic_folder = os.path.join(father_pic_folder, person)
            pic_list = os.listdir(pic_folder)
            if len(pic_list) < 80:
                continue
            url_file = os.path.join(father_file_folder, person+'.txt')
            if not os.path.exists(url_file):
                continue
            dest_folder = os.path.join(father_pic_box_folder, person)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            get_bound_box(pic_folder, url_file, dest_folder)
        except:
            # traceback.print_exc()
            continue

if __name__ == '__main__':
    main()