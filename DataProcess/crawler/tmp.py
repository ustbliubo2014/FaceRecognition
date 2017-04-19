# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/7/28 10:35
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import logging
from logging.config import fileConfig
import os
import re
import pdb
reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


for line in open('m1p50.html'):
    start_index = line.find('person')
    if start_index > 0:
        real_start = line[start_index:]
        m = re.match(r'.*person.*title=.*', line.rstrip())
        if m:
            pdb.set_trace()
            print m.group()
