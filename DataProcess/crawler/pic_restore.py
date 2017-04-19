# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: pic_restore.py
@time: 2016/7/27 14:12
@contact: ustb_liubo@qq.com
@annotation: pic_restore : 将hadoop下载的图片转换成png格式
"""
import sys
import logging
import os
import base64
from time import time
import pdb
import traceback
reload(sys)
sys.setdefaultencoding("utf-8")


logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='pic_restore.log',
                filemode='a+')


def restore(pic_str, pic_file_name):
    if 'error' in pic_str:
        return
    with open(pic_file_name, 'w') as f:
        pic = base64.b64decode(pic_str)
        f.write(pic)
        f.close()


def main_restore(hadoop_data_folder, all_pic_folder):
    '''
        :param hadoop_data_folder:hadoop文件的文件夹
        :param all_pic_folder:新生成的图片的文件夹
        :return:
    '''
    file_list = os.listdir(hadoop_data_folder)
    for file_name in file_list:
        absolute_path = os.path.join(hadoop_data_folder, file_name)
        count = 0
        start = time()
        name = ''
        for line in open(absolute_path):
            if line.startswith('person_name'):
                continue
            tmp = line.rstrip().split()
            try:
                if len(tmp) >= 3:
                    name = tmp[0].encode('gbk')
                    pic_index = tmp[1]
                    pic_content = tmp[2]
                    pic_folder = os.path.join(all_pic_folder, name)
                    if not os.path.exists(pic_folder):
                        os.makedirs(pic_folder)
                    pic_file_name = os.path.join(pic_folder, pic_index+'.png')
                    restore(pic_content, pic_file_name)
                    count += 1
            except:
                traceback.print_exc()
                continue
        end = time()
        print name, count, (end-start)



if __name__ == '__main__':
    hadoop_data_folder = '/data02/pic_download/pic_hadoop'
    all_pic_folder = '/data02/pic_download/pictures'
    main_restore(hadoop_data_folder, all_pic_folder)
