# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: stat.py
@time: 2016/7/25 10:22
@contact: ustb_liubo@qq.com
@annotation: stat
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='stat.log',
                    filemode='a+')


def stat(folder):
    folder_list = os.listdir(folder)
    people_count_dic = {}
    user_count = 0
    pic_num = 0
    for user_folder in folder_list:
        user_count += 1
        print user_count, len(people_count_dic)
        user_folder_list = os.path.join(folder, user_folder)
        pic_folder_list = os.listdir(user_folder_list)
        for pic_folder in pic_folder_list:
            pic_folder = os.path.join(folder, user_folder, pic_folder)
            if not os.path.isdir(pic_folder):
                continue
            pic_list = os.listdir(pic_folder)
            for pic in pic_list:
                if pic.endswith('.jpg'):
                    pic_num += 1
                    tmp = pic.split('_')
                    if len(tmp) == 2:
                        people_count_dic[tmp[0]] = people_count_dic.get(tmp[0], 0) + 1
    items = people_count_dic.items()
    items.sort(key=lambda x:x[1], reverse=True)
    pic_50 = pic_40 = pic_30 = pic_20 = pic_10 = 0
    for k in items:
        if k[1] >=50:
            pic_50 += 1
        if k[1] >= 40:
            pic_40 += 1
        if k[1] >= 30:
            pic_30 += 1
        if k[1] >= 20:
            pic_20 += 1
        if k[1] >= 10:
            pic_10 += 1
    print 'pic_num: ', pic_num, 'person_num :', len(people_count_dic), \
        'pic_50 :', pic_50, 'pic_40 :', pic_40, 'pic_30 :', pic_30, 'pic_20 :', pic_20, 'pic_10: ', pic_10


if __name__ == '__main__':
    folder = '/data/liubo/face/MegaFace_dataset/FlickrFinal2'
    stat(folder)
