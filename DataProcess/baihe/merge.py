# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: merge.py
@time: 2017/1/9 13:24
@contact: ustb_liubo@qq.com
@annotation: merge
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
    src_folder = '/data/liubo/face/baihe/person7w'
    dst_folder = '/data/liubo/face/baihe/all_person'
    folder_list = os.listdir(src_folder)
    for index, folder in enumerate(folder_list):
        start = time()
        src_folder_path = os.path.join(src_folder, folder)
        dst_folder_path = os.path.join(dst_folder, folder)
        if os.path.exists(dst_folder_path):
            continue
        else:
            shutil.move(src_folder_path, dst_folder_path)
            print index, time() - start
