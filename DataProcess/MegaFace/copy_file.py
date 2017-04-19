# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: copy_file.py
@time: 2016/7/25 11:05
@contact: ustb_liubo@qq.com
@annotation: copy_file : 将图片以人来分开(一个人的图片放在一个文件夹下)
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='copy_file.log',
                    filemode='a+')


def copy_file(old_folder, new_folder):
    folder_list = os.listdir(old_folder)
    people_path_dic = {}
    user_count = 0
    pic_num = 0
    for user_folder in folder_list:
        user_count += 1
        print user_count, len(people_path_dic)
        user_folder_list = os.path.join(old_folder, user_folder)
        pic_folder_list = os.listdir(user_folder_list)
        for pic_folder in pic_folder_list:
            pic_folder = os.path.join(old_folder, user_folder, pic_folder)
            if not os.path.isdir(pic_folder):
                continue
            pic_list = os.listdir(pic_folder)
            for pic in pic_list:
                if pic.endswith('.jpg'):
                    pic_num += 1
                    tmp = pic.split('_')
                    if len(tmp) == 2:
                        person = tmp[0]
                        pic_path = os.path.join(pic_folder, pic)
                        pic_json_path = os.path.join(pic_folder, pic+'.json')
                        if os.path.exists(pic_path) and os.path.exists(pic_json_path):
                            if person in people_path_dic:
                                pic_path_list = people_path_dic.get(person)
                            else:
                                pic_path_list = []
                            pic_path_list.append((pic_path, pic_json_path))
                            people_path_dic[person] = pic_path_list
    copy_num = 0
    for person in people_path_dic:
        pic_path_list = people_path_dic.get(person)
        if len(pic_path_list) >= 10:
            person_folder = os.path.join(new_folder, person)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)
            for e in pic_path_list:
                pic_path, pic_json_path = e
                shutil.copy(pic_path, person_folder)
                shutil.copy(pic_json_path, person_folder)
            copy_num += 1
            print copy_num


if __name__ == '__main__':
    old_folder = '/data/liubo/face/MegaFace_dataset/FlickrFinal2/'
    new_folder = '/data/liubo/face/MegaFace_dataset/people/'
    copy_file(old_folder, new_folder)