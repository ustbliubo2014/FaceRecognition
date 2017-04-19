# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: verif_gongka.py
@time: 2017/1/4 16:59
@contact: ustb_liubo@qq.com
@annotation: verif_gongka
"""
import sys
import os
from facepp import API, File
import pdb
import traceback
import requests
from time import sleep
import json

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


API_KEY = '156e3fc62553a80f0097805e994e3856'
API_SECRET = '5Ch01NIgILos90tdCAclStVGEnq6m5SJ'
api = API(API_KEY, API_SECRET)


def post_data(url):
    while True:
        try:
            request = requests.get(url)
            if request.status_code == 200:
                return json.loads(request.content).get('similarity')
        except:
            sleep(1)
            continue


def verif_two_face(faceid1, faceid2):
    url = 'https://apicn.faceplusplus.com/v2/recognition/compare?api_secret=%s&api_key=%s&face_id1=%s&face_id2=%s' \
          %(API_SECRET, API_KEY, faceid1, faceid2)
    # print url
    similarity = post_data(url)
    return similarity


def create_face_id(pic_folder, result_file):
    file_list = os.listdir(pic_folder)
    f_result = open(result_file, 'w')
    for file_name in file_list:
        try:
            file_path = os.path.join(pic_folder, file_name)
            face = api.detection.detect(img=File(file_path))
            face_id = face['face'][0]['face_id']
            f_result.write(file_path.decode('gbk')+'\t'+face_id+'\n')
            print file_path.decode('gbk')
        except:
            traceback.print_exc()
            continue


def verif_all_person(all_person_file, test_person_file):
    all_person_list = open(all_person_file).read().split('\n')
    test_person_list = open(test_person_file).read().split('\n')
    for test_element in test_person_list:
        test_tmp = test_element.split('\t')
        this_similarity_list = []
        if len(test_tmp) == 2:
            test_pic_path, test_face_id = test_tmp[0], test_tmp[1]
            for all_element in all_person_list:
                all_tmp = all_element.split('\t')
                if len(all_tmp) == 2:
                    all_pic_path, all_face_id = all_tmp[0], all_tmp[1]
                    similarity = float(verif_two_face(test_face_id, all_face_id))
                    # print similarity
                    this_similarity_list.append((similarity, all_pic_path))
                else:
                    # print 'all_tmp :', all_tmp
                    continue
        else:
            # print 'test_tmp :', test_tmp
            continue
        this_similarity_list.sort(key=lambda x:x[0])
        print test_tmp[0], test_tmp[1], this_similarity_list[-1][0], this_similarity_list[-1][1]


if __name__ == '__main__':
    # create_face_id(pic_folder='C:\Users\liubo\Desktop\picture\skyeye_pic',
    #                result_file = 'C:\Users\liubo\Desktop\picture/skyeye_pic_face_id.txt')
    # create_face_id(pic_folder='C:\Users\liubo\Desktop\picture/face++_test_pic',
    #                result_file = 'C:\Users\liubo\Desktop\picture/face++_test_pic_face_id.txt')
    # verif_all_person(all_person_file='C:\Users\liubo\Desktop\picture/skyeye_pic_face_id.txt',
    #                  test_person_file='C:\Users\liubo\Desktop\picture/face++_test_pic_face_id.txt')
    print verif_two_face('c0679a476eb8be84ab31148e5e64b254', '474689ca8571f6ddc5405c51988d3a9e')
    pass
