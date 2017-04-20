# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: sql_operator.py
@time: 2016/11/16 19:24
@contact: ustb_liubo@qq.com
@annotation: sql_operator : mysql相关操作
"""
import sys
import logging
from logging.config import fileConfig
import os
import MySQLdb
import pdb
import traceback
import time
import cv2
import numpy as np
import base64

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

def execute_query_sql(sql_operator):
    '''
        连接数据库, 执行语句, 关闭数据库
        :param sql_operator: 要执行的sql语句
        :return:
    '''
    db = MySQLdb.connect(host='10.16.66.44', port=3306, user='root', passwd='tianyan', db='face', charset='utf8')
    try:
        cursor = db.cursor()
        tmp = cursor.execute(sql_operator)
        result = cursor.fetchmany(tmp)
        db.close()
    except:
        if db != None:
            db.close()
        return None
    return result


def execute_update_sql(sql_operator):
    db = MySQLdb.connect(host='10.16.66.44', port=3306, user='root', passwd='tianyan', db='face', charset='utf8')
    cursor = db.cursor()
    try:
        cursor.execute(sql_operator)
        db.commit()
    except:
        traceback.print_exc()
        db.rollback()
    db.close()


def get_name_id(name):
    query_sql_operator = "select id from person where name = '%s'"%name
    result = execute_query_sql(query_sql_operator)
    return result


def insert_one_pic(person_id, img_str, half_str, algorithm, is_moved):
    insert_img_sql_operator = 'INSERT INTO images (person_id, ALGORITHM, half, img, is_moved ) ' \
                              'VALUES (%d, \'%s\', \'%s\', \'%s\', %d);'%(person_id, algorithm, half_str, img_str, is_moved)
    execute_update_sql(insert_img_sql_operator)


def insert_new_name(name, img_str):
    insert_person_sql_operator = 'INSERT INTO person (name, img) VALUES (\'%s\', \'%s\');' % (name, img_str)
    execute_update_sql(insert_person_sql_operator)
    name_id = get_name_id(name)
    return name_id


def insert_pic_list(pic_info_list):
    '''
        导入数据
    :param pic_info_list: [(name, algorithm, face_str, img_str), ..., (name, algorithm, face_str, img_str)]
    :return:
    '''
    all_name = get_all_name()
    name_id_dic = {}
    for k in all_name:
        this_id, this_name = k
        if 'new_person' in this_name:
            continue
        name_id_dic[this_name] = this_id
    for element in pic_info_list:
        try:
            name, algorithm, face_str, img_str = element
            name = name.encode('utf-8').decode('utf-8')
            print len(face_str), len(img_str)
            tmp_array  = cv2.imdecode(np.fromstring(base64.decodestring(img_str), dtype=np.uint8), 1)
            cv2.imwrite(str(time.time())+'.jpg', tmp_array)
            if name in name_id_dic:
                print 'has this person :', name
                person_id = name_id_dic.get(name)
                insert_one_pic(person_id, face_str, img_str, algorithm, is_moved=1)
            else:
                print 'no this person :', name
                person_id = insert_new_name(name, face_str)
                person_id = person_id[0][0]
                insert_one_pic(person_id, face_str, img_str, algorithm, is_moved=1)
                name_id_dic[name] = person_id
        except:
            traceback.print_exc()
            continue


def get_all_new_face():
    # 将数据库中所有新修改的人脸找出 -- 直接返回图片和该图片对应的人名
    query_sql_operator = 'select images.feature, person.name from images JOIN person where (images.is_moved = 1 and images.person_id = person.id)'
    # update_sql_operator = "UPDATE images SET is_moved = 2 WHERE is_moved = 1"
    result = execute_query_sql(query_sql_operator)
    # execute_update_sql(update_sql_operator)
    return result


def get_all_annotate_half():
    # 将数据库中所有新修改的人脸找出 -- 直接返回图片(半身照)和该图片对应的人名
    query_sql_operator = 'select images.half, person.name from images JOIN person where (images.is_moved = 1 and images.person_id = person.id)'
    result = execute_query_sql(query_sql_operator)
    return result


def get_all_name():
    # 找出数据库中所有人名
    query_sql_operator = "select id, name from person"
    result = execute_query_sql(query_sql_operator)
    return result


def download_all_images():
    query_sql_operator = "select id, img, half, feature from images where id < 1500000 and id > 1000000 and algorithm='institute'"
    result = execute_query_sql(query_sql_operator)
    return result


def clear_table():
    delete_sql_operator = "delete from person where id > 0"
    result = execute_update_sql(delete_sql_operator)
    delete_sql_operator = "delete from images where id > 0"
    result = execute_update_sql(delete_sql_operator)


def get_annotate_img():
    sql_operator = 'select images.img, person.name, images.feature, images.is_moved from ' \
                   'images JOIN person where (images.person_id = person.id)'
    result = execute_query_sql(sql_operator)
    # update_sql = 'update images set images.is_moved = 2 where (images.is_moved = 1)'
    # execute_update_sql(update_sql)
    return result


if __name__ == '__main__':
    pass

    # folder = 'show'
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # for element in result:
    #     image = element[0]
    #     image = base64.decodestring(image)
    #     im = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1)
    #     cv2.imwrite(os.path.join(folder, element[1].encode('gbk') + '.jpg'), im)
