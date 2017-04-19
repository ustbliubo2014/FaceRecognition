# coding=utf-8
import base64
import json
import sys
import os
import random

import urllib2, urllib
import json
import base64
import string
import random
import traceback
import os
from time import time
import base64
import json
import sys
import os
import random
import json
import pdb
from analyse_json import analyse_image_result,analyse_face_cluster_result
import msgpack


storm_url = 'http://drpc1v.cldisk2.tjhc.qihoo.net:3775/drpc/hdp-iri_api'

print storm_url

def strom_run(post_data):
    json_obj = json.loads(post_data)
    json_obj["debug"] = True
    post_data = json.dumps(json_obj)
    requestPOST = urllib2.Request(data=post_data, url=storm_url)
    requestPOST.get_method = lambda: 'POST'
    f = urllib2.urlopen(requestPOST)
    recog_rlt = f.read()
    return recog_rlt


def face_cluster_run(post_data):
    return strom_run(post_data)


def write_txt_lines( txt_file_name , list_str_lines ,with_end = False ):
    if (not with_end ):
        list_str_lines = [line+'\n' for line in list_str_lines]
    file_object = open(txt_file_name, 'w')
    file_object.writelines(list_str_lines)
    file_object.close()


def write_test_sh(txt_file_name , json_str):
    out_str = """storm_sync_send -idc bjdt hdp-iri_api '"""+json_str+"'"
    write_txt_lines(txt_file_name , [out_str])


def add_user():
    for i in range(0, 50):
        request = {
            "version":"",
            "method": "meta_add_user",
            "data":{
                "qid":"007",
            }
        }
        if i % 100 == 0:
            print i
        face_cluster_run(json.dumps(request))
    request = {
        "version":"",
        "method": "meta_get_user",
        "data":{
            "qid":"007",
        }
    }
    print 'face_cluster_run(json.dumps(request)) ', face_cluster_run(json.dumps(request))


def del_user():
    user_list = range(3126, 3176)
    for user in user_list:
        request = {
        "version":"",
        "method": "meta_del_user",
        "data":{
            "qid":"007",
            "user_id": user
            }
            }
        face_cluster_run(json.dumps(request))
    request = {
        "version":"",
        "method": "meta_get_user",
        "data":{
            "qid":"007",
        }
    }
    print 'face_cluster_run(json.dumps(request)) ', face_cluster_run(json.dumps(request))


def test_user():
    request = {
        "version":"",
        "method": "meta_get_user",
        "data":{
            "qid":"007",
        }
    }
    print 'face_cluster_run(json.dumps(request)) ', face_cluster_run(json.dumps(request))


def test_image(father_dir):

    face_id_list_dic = {}   # {name: face_id_list}
    user_id_list_dic = {}   # {name: user_id_list}
    count = 0
    start = time()
    person_list = os.listdir(father_dir)
    person_list.sort()

    # 先要获取训练图片的face_id
    for person in person_list[:]:

        person_path = os.path.join(father_dir, person)
        pic_list = os.listdir(person_path)
        pic_list.sort()

        for pic in pic_list:
            try:
                pic_path = os.path.join(person_path, pic)
                with open(pic_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                    request = {
                        "version": "",
                        "method": "get_face_info",
                        "data": {
                            "qid": "007",
                            "imagedata": encoded_string,
                            "image_id": pic_path,
                            "image_time": 999,
                            "face_id": pic_path
                        }
                    }
                    tmp = face_cluster_run(json.dumps(request))
                    image_id, face_id, user_id = analyse_image_result(tmp)
                    print 'image_id :', image_id, 'face_id :', face_id, 'user_id :', user_id
                    if image_id:
                        name = person
                        face_id_list = face_id_list_dic.get(name, [])
                        face_id_list.append(face_id)
                        face_id_list_dic[name] = face_id_list
                        user_id_list = user_id_list_dic.get(name, [])
                        user_id_list.append(user_id)
                        user_id_list_dic[name] = user_id_list
                    else:
                        name = person
                        user_id_list = user_id_list_dic.get(name, [])
                        user_id_list.append(user_id)
                        user_id_list_dic[name] = user_id_list
                    count += 1
            except:
                traceback.print_exc()
                continue

    # face_id_list_dic: {
    #    'zhangshunlong': [u'/data/liubo/face/self_train/zhangshunlong/zhangshunlong1468293557.74.png_face_0.jpg_low_0'],
    #    'lining': [u'/data/liubo/face/self_train/lining/lining1468293945.0.png_face_0.jpg_0'],}
    # user_id_list_dic:
    #           {'liulvzhou': [0], 'liuzhen': [0], 'huangwei': , 'zhangtianyu': [0], 'liuyang5-s': [0], 'lijun': [0]}
    return face_id_list_dic, user_id_list_dic


def test_face_pre_cluster():
    print "face_pre_cluster"
    request = {
    "version": "",
    "method": "face_pre_cluster",
    "data": {
        "qids": ["007"],
        "max_limit": 200,
        }
    }
    print face_cluster_run(json.dumps(request))


def test_face_cluster():
    print "get_face_cluster"
    request = {
        "version": "",
        "method":  "get_face_cluster",
        "data": {
            "qid": "007",
        }
    }
    json_data = face_cluster_run(json.dumps(request))
    analyse_face_cluster_result(json_data)


def test_remove_face():
    qid = "007"
    print "remove_face"
    request = {
    "version":"",
    "method": "remove_face",
        "data": {
            "qid":qid,
            'face_ids': ["24979971-36020144175-3-1448930773365.jpggVr_0"]
        }
    }
    print face_cluster_run(json.dumps(request))


def test_tag2(face_id_list_dic):
    qid = "007"
    print "test_tag2"
    current_user_id = 3176
    person_user_id_trans_dic = {}
    print 'all name : ', face_id_list_dic.keys()
    for name in face_id_list_dic:
        face_id_list = face_id_list_dic.get(name)
        request = {
            "version": "",
            "method": "tag_face",
            "data": {
                "qid": qid,
                "user_id": current_user_id,
                'face_ids': face_id_list
            }
        }
        person_user_id_trans_dic[name] = current_user_id
        current_user_id += 1
        face_cluster_run(json.dumps(request))
    return person_user_id_trans_dic


def test_de_tag():
    qid = "007"
    print "de_tag_face"
    request = {
    "version":"",
    "method": "de_tag_face",
        "data":{
            "qid":qid,
            "user_id": 5,
            'face_id': ["24979971-36020144175-3-1448934581509.jpg_0"]
        }
    }
    print face_cluster_run(json.dumps(request))


def test_getFace():
    qid = "007"
    print "get_face_orgin_data"
    request = {
        "version":"",
        "method": "get_face_orgin_data",
        "data":{
            "qid":qid,
            'face_id':"57075002e666c11d5f8b4577_0"
        }
    }
    print face_cluster_run(json.dumps(request))


def test_debug_set_cluster_threshold():
    print "debug_set_cluster_threshold"
    request = {
    "version":"",
    "method": "debug_set_cluster_threshold",
        "data":{
            'threshold':0.5
        }
    }
    print face_cluster_run(json.dumps(request))


def cal_acc(valid_user_id_list_dic, person_user_id_trans_dic):
    # 需要给出对应关系(这里手动写成dic)
    wrong_num = 0
    right_num = 0
    no_find = 0
    no_recognization = 0
    for name in valid_user_id_list_dic:
        user_id_list = valid_user_id_list_dic.get(name)
        right_id = person_user_id_trans_dic.get(name)
        # pdb.set_trace()
        for user_id in user_id_list:
            if user_id == -1 :
                no_find += 1
                continue
            if user_id == 0:
                no_recognization += 1
                continue
            if user_id != right_id:
                wrong_num += 1
            else:
                right_num += 1
            # pdb.set_trace()
    print 'right_num : ', right_num, 'wrong_num : ', wrong_num, 'no_find :', no_find, 'no_recognization :', \
        no_recognization, 'acc : ', (right_num*1.0/(right_num+wrong_num)*1.0)


def main_train():
    father_dir = '/data/liubo/face/self_train'
    train_face_id_list_dic, train_user_id_list_dic = test_image(father_dir)
    msgpack.dump((train_user_id_list_dic), open('train_user_id_list_dic.p', 'wb'))
    person_user_id_trans_dic = test_tag2(train_face_id_list_dic)
    msgpack.dump(person_user_id_trans_dic, open('person_user_id_trans_dic.p', 'wb'))

def main_valid():
    father_dir = '/data/liubo/face/self_valid'
    valid_face_id_list_dic, valid_user_id_list_dic = test_image(father_dir)
    msgpack.dump((valid_user_id_list_dic), open('test_user_id_list_dic.p', 'wb'))


def get_user():
    request = {
        "version":"",
        "method": "meta_get_user",
        "data":{
            "qid":"007",
        }
    }
    print 'face_cluster_run(json.dumps(request)) ', face_cluster_run(json.dumps(request))


if __name__ == '__main__':
    # get_user()
    # main_train()
    # main_valid()
    valid_user_id_list_dic = msgpack.load(open('test_user_id_list_dic.p', 'rb'))
    person_user_id_trans_dic = msgpack.load(open('person_user_id_trans_dic.p', 'rb'))
    cal_acc(valid_user_id_list_dic, person_user_id_trans_dic)
    print valid_user_id_list_dic

    # main_valid()
    # main_acc()
    # test_image('/data/liubo/face/self_train/')
    # del_user()
    # get_user()
    # add_user()
    # get_user()
