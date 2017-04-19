# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: MergeData.py
@time: 2016/8/22 14:42
@contact: ustb_liubo@qq.com
@annotation: MergeData : 将不同类型的数据合并在一起
    all_pic_data_folder:(只保存脸部数据)
        originalimages:
            person_A:
                pic_1
                ...
                pic_n
            ...
            person_M:
                pic_1
                ...
                pic_n
        ...
        FaceScrub:
            person_A:
                pic_1
                ...
                pic_n
            person_M:
                pic_1
                ...
                pic_n
"""
import sys
import shutil
import os
import caffe
from scipy.misc import imread, imresize
import numpy as np
from time import time
import pdb
import sklearn.metrics.pairwise as pw
from os.path import getsize


reload(sys)
sys.setdefaultencoding("utf-8")


merge_folder = '/data/liubo/face/all_pic_data'
same_threshold = 0.6    # 大于0.6的认为图片没有意义,删除
file_size_threshold = 5000    # 文件大小小于5000的图片没有意义, 直接删除


def read_one_rgb_pic(pic_path, pic_shape=(224, 224, 3)):
    img = imresize(imread(pic_path), pic_shape)
    return img


def copy_FaceScrub():
    FaceScrub_folder = '/data/liubo/face/FaceScrub/download'
    person_list = os.listdir(FaceScrub_folder)
    dst_folder = '/data/liubo/face/all_pic_data/FaceScrub'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for person in person_list:
        print person
        src_person_face_path = os.path.join(FaceScrub_folder, person, 'face')
        dst_person_face_path = os.path.join(dst_folder, person)
        shutil.copytree(src_person_face_path, dst_person_face_path)


def orl_data():
    orl_folder = '/data/liubo/face/face_DB/ORL/ORL_data'
    pic_list = os.listdir(orl_folder)
    for pic in pic_list:
        label = pic.split('_')[1]
        dst_folder = os.path.join(merge_folder, 'orl', label)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        pic_path = os.path.join(orl_folder, pic)
        shutil.copy(pic_path, dst_folder)


def originalimages():
    originalimages_folder = '/data/liubo/face/originalimages/data'
    pic_list = os.listdir(originalimages_folder)
    for pic in pic_list:
        label = pic.split('-')[0]
        dst_folder = os.path.join(merge_folder, 'original', label)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        pic_path = os.path.join(originalimages_folder, pic)
        shutil.copy(pic_path, dst_folder)


def filter_pic(person_folder):
    start = time()
    pic_list = os.listdir(person_folder)
    pic_path_feature_dic = {}  # {path:feature}
    for pic in pic_list:
        pic_path = os.path.join(person_folder, pic)
        if getsize(pic_path) < file_size_threshold:
            os.remove(pic_path)
            continue
        pic_feature = np.reshape(read_one_rgb_pic(pic_path, (64, 64, 3)), (1, (64*64*3)))
        pic_path_feature_dic[pic_path] = pic_feature
    print 'finish extract feature'
    path_list = pic_path_feature_dic.keys()
    del_pic_list = []
    for index_i, pic_path in enumerate(path_list):
        max_sim_score = 0
        for index_j, other_pic_path in enumerate(path_list[index_i+1:]):
            score = pw.cosine_similarity(pic_path_feature_dic.get(pic_path), pic_path_feature_dic.get(other_pic_path))
            if max_sim_score < score:
                max_sim_score = score
        if max_sim_score > same_threshold:
            del_pic_list.append(pic_path)
    end = time()
    print 'time :', (end - start)
    return del_pic_list


def vgg_data():
    # 对于每个人,将相似度很高的删除掉 --- 只保留前100张图片,不容易确定阈值
    vgg_folder = '/data/liubo/face/all_pic_data/vgg_data/'
    person_list = os.listdir(vgg_folder)
    for person in person_list:
        print person
        person_path = os.path.join(vgg_folder, person)
        pic_list = os.listdir(person_path)
        pic_list.sort()
        pic_list = map(lambda x:os.path.join(person_path, x), pic_list)
        count = 0
        for pic in pic_list:
            if getsize(pic) < file_size_threshold:
                os.remove(pic)
                continue
            else:
                count += 1
                if count > 100:
                    os.remove(pic)


def stat(folder):
    person_list = os.listdir(folder)
    pic_sum = 0
    for person in person_list:
        pic_sum = len(os.listdir(os.path.join(folder, person))) + pic_sum
    print folder, pic_sum


if __name__ == '__main__':
    # copy_FaceScrub()
    # orl_data()
    # originalimages()
    # person_folder = '/data/liubo/face/all_pic_data/vgg_data/Lori_Loughlin'
    # del_pic_list = filter_pic(person_folder)
    # pdb.set_trace()
    # vgg_data()

    # stat('/data/liubo/face/all_pic_data/annotate')
    # stat('/data/liubo/face/all_pic_data/FaceScrub')
    # stat('/data/liubo/face/all_pic_data/original')
    # stat('/data/liubo/face/all_pic_data/orl')
    # stat('/data/liubo/face/all_pic_data/vgg_data')
    # stat('/data/liubo/face/all_pic_data/yalefaces')
    # /data/liubo/face/all_pic_data/annotate 60676
    # /data/liubo/face/all_pic_data/FaceScrub 33047
    # /data/liubo/face/all_pic_data/original 2800
    # /data/liubo/face/all_pic_data/orl 400
    # /data/liubo/face/all_pic_data/vgg_data 262200
    # /data/liubo/face/all_pic_data/yalefaces 165
    # 不包含vgg : 97088 ; 包含vgg : 359288
    # person_num : 不包含vgg : 1424;  包含vgg : 4046
    pass
