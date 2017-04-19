# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: cluster.py
@time: 2017/1/20 14:58
@contact: ustb_liubo@qq.com
@annotation: cluster
"""
import sys
import logging
from logging.config import fileConfig
import os
from sql_operator import download_all_images
import base64
import cv2
import numpy as np
import pdb
import msgpack_numpy
from sklearn.metrics import pairwise
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)


def download_data():
    suffix = '_1000000'
    result = download_all_images()
    img_folder = 'test_cluster{}'.format(suffix)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    result_file = 'test_cluster{}.txt'.format(suffix)
    f_result = open(result_file, 'w')
    f_result.write('id' + '\t' + 'feature' + '\n')
    error_num = 0
    for element in result:
        try:
            id, img, half, feature = element
            face_array = cv2.imdecode(np.fromstring(base64.decodestring(img), dtype=np.uint8), 1)
            if face_array.shape[0] < 80 or face_array.shape[1] < 80:
                continue
            half_array = cv2.imdecode(np.fromstring(base64.decodestring(half), dtype=np.uint8), 1)
            cv2.imwrite(os.path.join(img_folder, str(id) + '_half.jpg'), half_array)
            cv2.imwrite(os.path.join(img_folder, str(id) + '_face.jpg'), face_array)
            # 特征使用方法
            # msgpack_numpy.loads(base64.b64decode(feature))
            f_result.write(str(id) + '\t' + feature + '\n')
        except:
            traceback.print_exc()
            error_num += 1
            continue
    print len(result), error_num


def cal_sim():
    sim_file = 'test_cluster.txt'
    all_content = open(sim_file, 'r').read().split('\n')
    error_num = 0
    all_num = 0
    del_num = 0
    del_list = []
    last_index = 1
    id_feature_dic = {}
    error_id = []
    for content in all_content:
        try:
            id, feature = content.split('\t')
            id_feature_dic[id] = msgpack_numpy.loads(base64.b64decode(feature))
        except:
            error_id.append(id)
            continue
    # pdb.set_trace()
    for index in range(2, len(all_content)):
        try:
            last_id, last_feature = all_content[last_index].split('\t')
            this_id, this_feature = all_content[index].split('\t')
            this_feature = msgpack_numpy.loads(base64.b64decode(this_feature))
            last_feature = msgpack_numpy.loads(base64.b64decode(last_feature))
            if this_feature.size == 256 and last_feature.size == 256:
                cos_sim = pairwise.cosine_similarity(this_feature, last_feature)[0][0]
                all_num += 1
                if cos_sim > 0.85:
                    del_num += 1
                    del_list.append(this_id)
                    print this_id, last_id, last_index, cos_sim
                else:
                    last_index = index
        except:
            error_num += 1
            continue
    return del_list


def del_file():
    del_list = set(map(int, cal_sim()))
    print len(del_list)
    folder = 'test_cluster'
    pic_list = os.listdir(folder)
    for pic in pic_list:
        id = int(pic.split('_')[0])
        if id in del_list:
            print 'del'
            os.remove(os.path.join(folder, pic))


if __name__ == '__main__':
    download_data()
    # cal_sim()
    # del_file()
    # pass
