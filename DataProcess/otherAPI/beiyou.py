# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: beiyou.py
@time: 2016/8/9 17:30
@contact: ustb_liubo@qq.com
@annotation: beiyou
"""
import sys
import logging
from logging.config import fileConfig
import os
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

import requests # 需要单独安装
result = requests.post('http://api.bupt-search.com:81/v2/detection/detect',
                        data={'api_key': 'KUYVaAd06zoWfBFS2M9N6qfGKgVS1v7Ckx1cPnhQ',
                              'api_secret': '4uSTnMDZqXqurWa4JwLT6YsSti4cth6R0K0JcdsL'},
                        files={'img_file': open('huangchuanming1468293383.83.png', 'rb')})
print result
pdb.set_trace()