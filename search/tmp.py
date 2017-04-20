# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/7/13 16:03
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='tmp.log',
                    filemode='a+')


if __name__ == '__main__':
    vgg_folder = '/data/liubo/face/lfw_face'
    new_vgg_folder = '/data/liubo/face/lfw_face_new'
    os.makedirs(new_vgg_folder)
    person_list = os.listdir(vgg_folder)
    num = 0
    for person in person_list:
        pic_list = os.listdir(os.path.join(vgg_folder, person))
        if len(pic_list) < 3:
            continue
        shutil.copytree(os.path.join(vgg_folder, person), os.path.join(new_vgg_folder, person))
        num += 1
    print num