# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: tmp.py
@time: 2017/2/7 15:05
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import logging
from logging.config import fileConfig
import os
import shutil
import pdb
import traceback
from time import time
import cv2
import pdb
import numpy as np
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':
    pass
    folder = sys.argv[1]
    person_list = os.listdir(folder)
    np.random.shuffle(person_list)
    count = 0
    for person in person_list:
        start = time()
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            try:
                pic_path = os.path.join(person_path, pic)
                img_array = cv2.imread(pic_path)
                if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    print 'error_pic :', pic_path
                # pic_path = os.path.join(person_path, pic)
                # new_pic_path = pic_path[:-3]+'png'
                # if pic_path.endswith('jpg'):
                #     img_array = cv2.imread(pic_path)
                #     if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                #         print 'error_pic :', pic_path
                #         continue
                #     cv2.imwrite(new_pic_path, img_array)
                #     os.remove(pic_path)
                # elif pic_path.endswith('png'):
                #     continue
                # else:
                #     print 'error :', pic_path
            except:
                traceback.print_exc()
                continue
        end = time()
        count += 1
        print 'count :', count, (end - start)

