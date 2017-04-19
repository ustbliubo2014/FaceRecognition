# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: crop_face.py
@time: 2017/1/5 11:45
@contact: ustb_liubo@qq.com
@annotation: crop_face : 利用检测结果得到人脸框(只要size>70的图片)
"""
import cv2
import os
import base64
import urllib
import urllib2
import json
from datetime import datetime
from time import time
import traceback
import zlib
import sys
import pdb
import requests
import shutil

reload(sys)
sys.setdefaultencoding("utf-8")


detect_url = "http://10.160.164.25:9999/"


def detect_face(img):
    try:
        request = {
            "image_id": "czc_test",
            "image": base64.encodestring(((cv2.imencode('.jpg', img)[1].tostring())))
        }
        result = requests.post(detect_url, data=request)
        face_pos = json.loads(result.content)['detection']
        face_num = len(face_pos)
        if face_num > 1:
            return None
        for index in range(len(face_pos)):
            try:
                x, y, w, h = face_pos[index]
                center_x = x + w / 2
                center_y = y + h / 2
                new_x_min = int(max(center_x - w * 0.5, 0))
                new_x_max = int(min(center_x + w * 0.5, img.shape[1]))
                new_y_min = int(max(center_y - h * 0.5, 0))
                new_y_max = int(min(center_y + h * 0.5, img.shape[0]))
                face_array = img[new_y_min:new_y_max, new_x_min:new_x_max]
                return face_array
            except:
                traceback.print_exc()
                continue
        return None
    except:
        traceback.print_exc()
        return None


if __name__ == '__main__':
    src_folder = '/data/liubo/face/baihe/person'
    dst_folder = '/data/liubo/face/baihe/person_face'
    person_list = os.listdir(src_folder)
    person_index = 1
    size_threshold = 70
    train_resize = 182
    start = time()
    for person in person_list:
        person_index += 1
        src_person_path = os.path.join(src_folder, person)
        dst_person_path = os.path.join(dst_folder, person)
        if not os.path.exists(dst_person_path):
            os.makedirs(dst_person_path)
        pic_list = os.listdir(src_person_path)
        for pic in pic_list:
            try:
                src_pic_path = os.path.join(src_person_path, pic)
                dst_pic_path = os.path.join(dst_person_path, pic)
                if os.path.exists(dst_pic_path):
                    continue
                src_img = cv2.imread(src_pic_path)
                dst_img = detect_face(src_img)
                if dst_img == None:
                    continue
                if dst_img.shape[0] < size_threshold or dst_img.shape[1] < size_threshold:
                    continue
                if src_img == None:
                    continue
                else:
                    cv2.imwrite(dst_pic_path, cv2.resize(dst_img, (train_resize, train_resize)))
            except:
                traceback.print_exc()
                continue
        print person_index, person, (time() - start), len(pic_list)
        start = time()