# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: filter_data.py
@time: 2016/7/18 10:20
@contact: ustb_liubo@qq.com
@annotation: filter_data
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='filter_data.log',
                    filemode='a+')

'''
    训练只选取所有train的10%训练
'''
import msgpack
from random import randint
import pdb

def filter_data():
    sample_list_file = '/data/liubo/face/vgg_face_dataset/all_data/all_sample_list.p'
    train_list, valid_list = msgpack.load(open(sample_list_file,'rb'))
    small_train_list = []
    label_set = set()
    for element in train_list:
        if randint(0,9) == 0:
            small_train_list.append(element)
            label_set.add(element[1])
    if len(label_set) == 2622:
        msgpack.dump((small_train_list, valid_list), open('/data/liubo/face/vgg_face_dataset/all_data/sub_sample_list.p','wb'))

def part_person_data():
    sample_list_file = '/data/liubo/face/vgg_face_dataset/all_data/all_sample_list.p'
    train_list, valid_list = msgpack.load(open(sample_list_file,'rb'))
    small_train_list = []
    small_valid_list = []
    label_set = set()
    for element in train_list:
        if element[1] < 300:
            small_train_list.append(element)
            label_set.add(element[1])
    for element in valid_list:
        if element[1] < 300:
            small_valid_list.append(element)

    msgpack.dump((small_train_list, valid_list), open('/data/liubo/face/vgg_face_dataset/all_data/300person_sample_list.p','wb'))

if __name__ == '__main__':
    part_person_data()

