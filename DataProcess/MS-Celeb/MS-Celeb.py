# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: MS-Celeb.py
@time: 2016/9/1 14:48
@contact: ustb_liubo@qq.com
@annotation: MS-Celeb
"""
import sys
import logging
from logging.config import fileConfig
import os
import pdb
import base64
from time import time
import shutil
import traceback
import struct

reload(sys)
sys.setdefaultencoding("utf-8")


def extract():
    fid = open("/data/liubo/face/MS-Celeb_face/MsCelebV1-Faces-Cropped.tsv", "r")
    result_folder = '/data/liubo/face/MS-Celeb_face/MsCelebV1-Faces-Cropped'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    while True:
        line = fid.readline()
        if line:
            # 0: Freebase MID (unique key for each entity)
            # 1: ImageSearchRank
            # 4: FaceID
            # 5: bbox
            # 6: img_data
            data_info = line.split('\t')
            mid = data_info[0]
            person_folder = os.path.join(result_folder, mid)
            print person_folder
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)
            filename = person_folder + "/" + data_info[4] + "_" + data_info[1] + ".jpg"
            bbox = struct.unpack('ffff', data_info[5].decode("base64"))
            img_data = data_info[6].decode("base64")
            open(filename, 'w').write(img_data)
        else:
            break

    fid.close()


def rename():
    folder = '/data/liubo/face/MS-Celeb_face/MsCelebV1-Faces-Cropped'
    person_list = os.listdir(folder)
    for person_index, person in enumerate(person_list):
        print person_index, person
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            try:
                tmp = pic.split('.')
                tmp1 = tmp[0].split('_')
                new_pic_name = tmp1[1]+'-'+tmp1[0]+'.'+tmp[1]
                # FaceId-0_3.jpg
                src_pic_path = os.path.join(person_path, pic)
                dst_pic_path = os.path.join(person_path, new_pic_name)
                # print dst_pic_path
                shutil.move(src_pic_path, dst_pic_path)
            except:
                traceback.print_exc()
                continue


def clean_data():
    # 根据clean_list构建新的数据集
    folder = '/data/liubo/face/MS-Celeb_face/MsCelebV1-Faces-Cropped'
    txt_file = '/data/liubo/face/MS-Celeb_face/MS-Celeb-1M_clean_list.txt'

if __name__ == '__main__':
    rename()