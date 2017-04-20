# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: train_batch_model.py
@time: 2016/8/4 15:53
@contact: ustb_liubo@qq.com
@annotation: train_batch_model (数据很大时分批读入, 传入的是一个列表)
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

if __name__ == '__main__':
    pass
