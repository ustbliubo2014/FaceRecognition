# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: classify.py
@time: 2016/7/15 11:19
@contact: ustb_liubo@qq.com
@annotation: classify
"""
import sys
from cluster import classify
reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='classify.log',
                    filemode='a+')

if __name__ == '__main__':
    classify(data_folder='/data/liubo/face/vgg_face_dataset/train_valid', pic_num_threshold=10)
