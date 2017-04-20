# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: conf.py
@time: 2016/8/18 11:07
@contact: ustb_liubo@qq.com
@annotation: conf
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

originalimages_conv3_data_path = '/data/liubo/face/picture_feature/originalimages_conv3.p'
originalimages_verif_conv3_data_path = '/data/liubo/face/picture_feature/originalimages_verif_conv3.p'

originalimages_fc7_data_path = '/data/liubo/face/picture_feature/originalimages_fc7.p'
originalimages_verif_fc7_path_feature = '/data/liubo/face/picture_feature/originalimages_verif_fc7_path_feature.p'

orl_fc7_data_path = '/data/liubo/face/picture_feature/orl_fc7.p'
orl_verif_fc7_path_feature = '/data/liubo/face/picture_feature/orl_verif_fc7_path_feature.p'
if __name__ == '__main__':
    pass
