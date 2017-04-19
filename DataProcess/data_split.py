# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: data_split.py
@time: 2017/2/20 15:22
@contact: ustb_liubo@qq.com
@annotation: data_split : 将数据集分成多个小数据集, 方便上传到云盘
"""
import sys
import logging
from logging.config import fileConfig
import os
import shutil
from time import time

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    src_folder = '/data/liubo/face/MS-Celeb_face/clean_data'
    src_subfolder_list = os.listdir(src_folder)
    dst_folder = '/data/liubo/face/MS-Celeb_face/clean_data_split'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for index in range(0, len(src_subfolder_list), 1000):
        start = time()
        sub_list = src_subfolder_list[index:index+1000]
        for subfolder in sub_list[:1000]:
            # shutil.move(os.path.join(src_folder, subfolder), os.path.join(dst_folder, subfolder))
            shutil.move(os.path.join(src_folder, subfolder), dst_folder)
        os.system('tar -cvf /data/liubo/face/MS-Celeb_face/clean_data_split_{}.tgz '
                  '/data/liubo/face/MS-Celeb_face/clean_data_split'.format(index))
        os.system('rm -rf /data/liubo/face/MS-Celeb_face/clean_data_split/*')
        end = time()
        print 'time :', index, (end - start)
