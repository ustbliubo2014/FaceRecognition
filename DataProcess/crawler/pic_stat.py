# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: pic_stat.py
@time: 2016/7/27 15:12
@contact: ustb_liubo@qq.com
@annotation: pic_stat
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
    # 统计一个有多少图片
    folder = '/data02/pic_download/pictures'
    person_list = os.listdir(folder)
    pic_num = 0
    for person in person_list:
        try:
            pic_list = os.listdir(os.path.join(folder, person))
            pic_num += len(pic_list)
        except:
            import traceback
            traceback.print_exc()
        print person, pic_num
