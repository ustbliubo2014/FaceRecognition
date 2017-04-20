# encoding: utf-8
__author__ = 'liubo'

"""
@version:
@author: 刘博
@license: Apache Licence
@contact: ustbliubo@qq.com
@software: PyCharm
@file: tmp.py
@time: 2016/6/6 23:02
"""

import os
import shutil
import numpy as np
from optparse import OptionParser
import pdb

# parser = OptionParser()
# parser.add_option("-a", "--augment", dest="need_augment", help="weight file")
# (options, args) = parser.parse_args()
# if options.need_augment.rstrip() == 'True':
#     print 'augment'
# else:
#     print 'not augment'

import shutil

folder = 'C:\Users\liubo\Desktop\人脸识别\工卡照片\第1批'
new_folder = 'C:\Users\liubo\Desktop\人脸识别\工卡照片/all_pic'
part_list = os.listdir(folder)
pic_num_dic = {}
count = 0
all_pic_num = 0
for part in part_list:
    part_path = os.path.join(folder, part)
    pic_list = map(lambda x:os.path.join(part_path, x), os.listdir(part_path))
    for pic in pic_list:
        shutil.copy(pic, new_folder)
