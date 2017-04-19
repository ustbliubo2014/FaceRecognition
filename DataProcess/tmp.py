# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/7/27 12:06
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='tmp.log',
                    filemode='a+')

# value={'green':'hello world', 'language':'python'}
# print '%(green)s from %(language)s' % value

folder = '/data/liubo/face/baihe/train/person_dlib_face'
person_list = os.listdir(folder)
count = 0
for person in person_list:
    pic_list = os.listdir(os.path.join(folder, person))
    for pic in pic_list:
        if pic.endswith('jpg'):
            continue
        else:
            print os.path.join(folder, person, pic)
    count += 1
    if count % 1000 == 0:
        print count