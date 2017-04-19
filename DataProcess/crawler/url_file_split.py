# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: url_file_split.py
@time: 2016/8/1 16:44
@contact: ustb_liubo@qq.com
@annotation: url_file_split : 将url文件分开
"""
import sys
import logging
from logging.config import fileConfig
import os

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def url_file_split():
    folder = 'person_url_check'
    new_folder = 'person_url_check_new'
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    all_content_list = []
    file_list = map(lambda x:os.path.join(folder, x), os.listdir(folder))
    for file_name in file_list:
        all_content_list.extend(open(file_name).read().split('\n'))
    file_count = 0
    f = open(os.path.join(new_folder, str(file_count)), 'w')
    for index in range(len(all_content_list)):
        if (index+1) % 20 == 0:
            f.close()
            file_count += 1
            f = open(os.path.join(new_folder, str(file_count)), 'w')
            f.write(all_content_list[index].rstrip()+'\n')
        else:
            f.write(all_content_list[index].rstrip()+'\n')


if __name__ == '__main__':
    url_file_split()
