#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: test_recognize_server.py
@time: 2016/6/13 16:10
@contact: ustb_liubo@qq.com
@annotation: test_recognize_server
"""

import os
import random
from shutil import copyfile
import urllib2
from time import time
import base64
import urllib
from scipy.misc import imread, imresize
import json
import traceback
import numpy as np
import zlib
import pdb
import cv2
import msgpack

pic_shape = (224, 224)
port = 8888


def split_data(raw_folder='/data/liubo/face/self', train_folder='/data/liubo/face/self_train',
               valid_folder='/data/liubo/face/self_valid'):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)
    person_list = os.listdir(raw_folder)
    np.random.shuffle(person_list)
    person_list = person_list[:100]
    for person in person_list:
        raw_person_path = os.path.join(raw_folder, person)
        train_person_path = os.path.join(train_folder, person)
        valid_person_path = os.path.join(valid_folder, person)
        if not os.path.exists(train_person_path):
            os.makedirs(train_person_path)
        if not os.path.exists(valid_person_path):
            os.makedirs(valid_person_path)
        pic_list = os.listdir(raw_person_path)
        np.random.shuffle(pic_list)
        pic_list = pic_list[:50]
        for pic in pic_list:
            raw_pic_path = os.path.join(raw_person_path, pic)
            train_pic_path = os.path.join(train_person_path, pic)
            valid_pic_path = os.path.join(valid_person_path, pic)
            if random.randint(1, 10) <= 5:
                copyfile(raw_pic_path, train_pic_path)
            else:
                copyfile(raw_pic_path, valid_pic_path)


def valid_recognize(valid_folder):
    person_list = os.listdir(valid_folder)
    right_num = 0
    wrong_num = 0
    no_recognize_num = 0
    all_num = 0
    for person in person_list:
        person_path = os.path.join(valid_folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            face = imresize(imread(pic_path), pic_shape)
            img_str = zlib.compress(face.tostring())
            print 'type(img_str) :', type(img_str)
            request = {
                "image_id": time(),
                "request_type": 'recognization',
                # "image": base64.encodestring(face.tostring())
                "image": base64.encodestring(img_str)
            }
            requestPOST = urllib2.Request(
                data=urllib.urlencode(request),
                url="http://10.160.164.26:%d/"%port
            )
            requestPOST.get_method = lambda : "POST"
            try:
                s = urllib2.urlopen(requestPOST).read()
            except urllib2.HTTPError, e:
                print e.code
            except urllib2.URLError, e:
                print str(e)
            try:
                person_name, score_proba, save_person_num, need_save = json.loads(s)["recognization"]
            except:
                traceback.print_exc()
                all_num += 1
                no_recognize_num += 1
                person_name, score_proba = 'unknown', 1.0
            end = time()
            all_num += 1
            if person_name == '':
                no_recognize_num += 1
            elif person_name.startswith('new_person'):
                continue
            elif person == person_name:
                right_num += 1
            else:
                wrong_num += 1
            print  person, person_name, score_proba
    print all_num, right_num, wrong_num, no_recognize_num


def valid_one_pic_recognize(pic_path):
    # face = cv2.resize(cv2.imread(pic_path), pic_shape)
    # img_str = zlib.compress(cv2.imencode('.jpg', cv2.resize(face, (224, 224),
    # interpolation=cv2.INTER_LINEAR))[1].tostring())

    face = cv2.imread(pic_path)
    img_str = zlib.compress(cv2.imencode('.jpg', face)[1].tostring())

    request = {
        "image_id": '.'.join(os.path.split(pic_path)[1].split('.')[:-1]),
        "request_type": 'recognization',
        "image": base64.encodestring(img_str)
    }
    requestPOST = urllib2.Request(
        data=urllib.urlencode(request),
        url="http://10.160.164.26:%d/"%port
    )
    requestPOST.get_method = lambda : "POST"
    try:
        s = urllib2.urlopen(requestPOST).read()
    except urllib2.HTTPError, e:
        print e.code
    except urllib2.URLError, e:
        print str(e)
    try:
        person_name, score_proba, save_person_num, need_save = msgpack.loads(base64.b64decode(json.loads(s)["recognization"]))
        return person_name
    except:
        traceback.print_exc()
        person_name, score_proba = 'unknown', 1.0
    print person_name, score_proba


def valid_add(pic_path, person):
    # face = read_one_rgb_pic(pic_path)
    # 请求本地服务
    request = {
        "label": person,
        "request_type": 'add',
        "one_pic_feature": pic_path
    }
    requestPOST = urllib2.Request(
        data=urllib.urlencode(request),
        url="http://10.160.164.26:%d/"%port
    )
    requestPOST.get_method = lambda : "POST"
    try:
        s = urllib2.urlopen(requestPOST).read()
    except urllib2.HTTPError, e:
        print e.code
    except urllib2.URLError, e:
        print str(e)
    try:
        add_flag = json.loads(s)["add"]
        if not add_flag:    # 加载失败
            print 'no add file :', pic_path
    except:
        print 'no add file :', pic_path
        traceback.print_exc()


def add_all_capture():
    folder = '/data/liubo/face/capture'
    person_list = os.listdir(folder)
    for person in person_list:
        # if person != 'liubo-it':
        #     continue
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            print person, pic
            pic_path = os.path.join(person_path, pic)
            valid_add(pic_path, person)


if __name__ == '__main__':
    # pass
    # add_all_capture()
    pic_folder = '/tmp/2016-11-14-15-40'
    pic_list = map(lambda x:os.path.join(pic_folder, x), os.listdir(pic_folder))
    for pic in pic_list:
        start = time()
        person_name = valid_one_pic_recognize(pic)
        print os.path.split(pic)[1], person_name, (time()-start)


