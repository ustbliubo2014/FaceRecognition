# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: file_name_trans.py
@time: 2016/7/27 11:04
@contact: ustb_liubo@qq.com
@annotation: file_name_trans
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='file_name_trans.log',
                    filemode='a+')


# 将中文文件名转换成拼音
from xpinyin import Pinyin
p = Pinyin()
import os
folder = 'url'
new_folder = 'url_pinyin'

file_list = os.listdir(folder)
import shutil
for file_name in file_list:
    old_file_path = os.path.join(folder, file_name)
    new_file_path = os.path.join(new_folder, p.get_pinyin(unicode(file_name,'utf-8')).replace('-',''))
    shutil.copyfile(old_file_path, new_file_path)

