# encoding: utf-8
__author__ = 'liubo'

"""
@version: 
@author: 刘博
@license: Apache Licence 
@contact: ustb_liubo@qq.com
@software: PyCharm
@file: create_vgg_pair.py
@time: 2016/7/18 22:23
"""

import logging
import os

if not os.path.exists('log'):
    os.mkdir('log')

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/create_vgg_pair.log',
                    filemode='w')


def func():
    pass


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    pair_file = 'vgg_pair.txt'
    f = open(pair_file, 'w')
    folder = '/data/liubo/face/vgg_face_dataset/all_data/pictures_box'
    person_list = os.listdir(folder)
    length = len(person_list)
    for index, person in enumerate(person_list):
        next_index = (index + 1) % length
        f.write(person+'\t'+'0'+'\t'+person_list[next_index]+'\t'+'0'+'\n')
        f.write(person+'\t'+'0'+'\t'+'1'+'\n')
    f.close()
