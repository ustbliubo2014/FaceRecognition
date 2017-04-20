# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: test_server.py
@time: 2016/10/17 14:57
@contact: ustb_liubo@qq.com
@annotation: test_server
"""
import sys
import logging
from logging.config import fileConfig
import os
import json
import cv2
import urllib
import urllib2
import zlib
import base64
import traceback
import msgpack_numpy
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


check_port =7777


def valid_one_pic_pose(face_img, image_id):

    face_img_str = base64.b64encode(msgpack_numpy.dumps(face_img))

    request = {
        "request_type": 'check_pose',
        "face_img_str": face_img_str,
        "image_id": image_id,
    }

    requestPOST = urllib2.Request(
        data=urllib.urlencode(request),
        url="http://10.160.164.26:%d/" % check_port
    )
    requestPOST.get_method = lambda: "POST"
    try:
        s = urllib2.urlopen(requestPOST).read()
    except urllib2.HTTPError, e:
        print e.code
    except urllib2.URLError, e:
        print str(e)
    try:
        pose_predict = json.loads(s)["pose_predict"]
        if not pose_predict:  # 加载失败
            print image_id, 'pose filter'
            return False
        else:
            pose_predict = msgpack_numpy.loads(base64.b64decode(pose_predict))
            print pose_predict
            return True
    except:
        traceback.print_exc()
        return False


if __name__ == '__main__':
    face_img = cv2.imread('xiejunping1468293619.94.png_face_0.jpg')
    valid_one_pic_pose(face_img, 'test')
