# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: extract_feature.py
@time: 2016/8/17 19:09
@contact: ustb_liubo@qq.com
@annotation: extract_feature
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import msgpack_numpy
from util import load_model, read_one_rgb_pic
import pdb


avg = np.array([129.1863, 104.7624, 93.5940])


def extract(pic_path, get_Conv_FeatureMap, pic_shape):
    img = read_one_rgb_pic(pic_path, pic_shape, avg)
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    return feature_vector



