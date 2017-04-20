# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: lfw_data_process.py
@time: 2016/7/21 18:35
@contact: ustb_liubo@qq.com
@annotation: lfw_data_process
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
                    filename='lfw_data_process.log',
                    filemode='a+')

if __name__ == '__main__':
    lfw_face_folder = '/data/liubo/face/lfw_face'
    person_list = os.listdir(lfw_face_folder)
    lfw_face_more = '/data/liubo/face/lfw_face_more'
    os.makedirs(lfw_face_more)
    for person in person_list:
        if len(os.listdir(os.path.join(lfw_face_folder, person))) > 1:
            print person
            shutil.copytree(os.path.join(lfw_face_folder, person), os.path.join(lfw_face_more, person))
