# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: create_valid_pair.py
@time: 2016/8/10 18:13
@contact: ustb_liubo@qq.com
@annotation: create_valid_pair
"""
import sys
import logging
from logging.config import fileConfig
import os
from random import randint
import pdb
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

sample_num = 10 # 每个人10张正样本,10张负样本
same_person_id = 0
no_same_person_id = 1

def create_valid_pair(folder, pair_file):
    '''
    :param folder:
            folder
                personA
                    pic1
                    pic2
                personB
                    pic1
                    pic2
    :param pair_file:
        pic_path1 pic_path2 label
    :return:
    '''
    person_list = os.listdir(folder)
    person_path_dic = {}
    for person in person_list:
        person_path_dic[person] = \
            map(lambda x:os.path.join(os.path.join(folder, person), x), os.listdir(os.path.join(folder, person)))
    pair_list = []
    person_num = len(person_path_dic)
    for person in person_path_dic:
        try:
            this_person_path_list = person_path_dic.get(person)
            path_num = len(this_person_path_list)
            if path_num < 10:
                continue
            count = 0
            # 找10张不一样的
            while count < sample_num:
                other_person = person_list[randint(0, person_num-1)]
                if other_person == person:
                    continue
                other_person_path = person_path_dic.get(other_person)
                if len(other_person_path) < 1 or len(this_person_path_list) < 1:
                    continue
                count += 1
                pair_list.append((
                    this_person_path_list[randint(0, path_num-1)],
                    other_person_path[randint(0, len(other_person_path)-1)],
                    no_same_person_id
                    ))
            # 找10张一样的
            for index_i in range(1, 11):
                pair_list.append((
                        this_person_path_list[index_i],
                        this_person_path_list[0],
                        same_person_id
                ))
        except:
            traceback.print_exc()
            pdb.set_trace()
    f = open(pair_file, 'w')
    for element in pair_list:
        f.write('\t'.join(map(str, element))+'\n')
    f.close()


def create_self_valid_pair(folder, pair_file):
    # 实现的功能类似, 只是self的数据很少,所以要找出所有可能的正样本和对应数量的负样本
    person_list = os.listdir(folder)
    person_path_dic = {}
    for person in person_list:
        person_path_dic[person] = \
            map(lambda x:os.path.join(os.path.join(folder, person), x), os.listdir(os.path.join(folder, person)))
    pair_list = []
    person_num = len(person_path_dic)
    for person in person_path_dic:
        try:
            this_person_path_list = person_path_dic.get(person)
            path_num = len(this_person_path_list)
            count = 0

            sample_num = path_num * (path_num - 1) / 2
            while count < sample_num:
                other_person = person_list[randint(0, person_num-1)]
                if other_person == person:
                    continue
                other_person_path = person_path_dic.get(other_person)
                if len(other_person_path) < 1 or len(this_person_path_list) < 1:
                    continue
                count += 1
                pair_list.append((
                    this_person_path_list[randint(0, path_num-1)],
                    other_person_path[randint(0, len(other_person_path)-1)],
                    no_same_person_id
                    ))

            for index_i in range(0, path_num):
                for index_j in range(index_i+1, path_num):
                    pair_list.append((
                        this_person_path_list[index_i],
                        this_person_path_list[index_j],
                        same_person_id
                    ))
        except:
            traceback.print_exc()
            pdb.set_trace()
    f = open(pair_file, 'w')
    for element in pair_list:
        f.write('\t'.join(map(str, element))+'\n')
    f.close()


if __name__ == '__main__':
    # folder = '/data/hanlin/'
    # pair_file = '/data/verif_list.txt'
    # create_valid_pair(folder, pair_file)
    folder = '/data/liubo/face/tmp_verif'
    pair_file = '/data/liubo/face/tmp_verif_list.txt'
    create_self_valid_pair(folder, pair_file)