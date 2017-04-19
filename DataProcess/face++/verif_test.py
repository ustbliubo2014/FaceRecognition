# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: verif_test.py
@time: 2016/8/19 17:11
@contact: ustb_liubo@qq.com
@annotation: verif_test : 获取人脸的face_id并计算pair的相似度
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
API_KEY = '156e3fc62553a80f0097805e994e3856'
API_SECRET = '5Ch01NIgILos90tdCAclStVGEnq6m5SJ'
api = API(API_KEY, API_SECRET)


def get_face_id():
    folder = 'C:\Users\liubo\Desktop\picture\skyeye_pic'
    person_list = os.listdir(folder)
    f = open('skyeye_pic_path_faceid.txt', 'w')
    has_process_path = {}
    for line in open('pic_path_faceid.txt'):
        tmp = line.rstrip().split()
        if len(tmp) == 2:
            pic_path = tmp[0]
            face_id = tmp[1]
            if face_id == 'no_face':
                continue
            else:
                has_process_path[pic_path] = face_id
    print len(has_process_path)

    for person in person_list:
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            try:
                pic_path = os.path.join(person_path, pic)
                if pic_path in has_process_path:
                    continue
                result = api.detection.detect(img = File(pic_path))
                face = result.get('face')
                if len(face) > 0:
                    face_id = result.get('face')[0].get('face_id')
                    print pic_path, face_id
                    f.write(pic_path+'\t'+face_id+'\n')
                else:
                    print pic_path, 'no_face'
                    f.write(pic_path+'\t'+'no_face'+'\n')
            except:
                traceback.print_exc()
                pdb.set_trace()


def verif_two_face(faceid1, faceid2):
    url = 'https://apicn.faceplusplus.com/v2/recognition/compare?api_secret=%s&api_key=%s&face_id1=%s&face_id2=%s' \
          %(API_SECRET, API_KEY, faceid1, faceid2)
    similarity = post_data(url)
    return similarity


def post_data(url):
    while True:
        try:
            request = requests.get(url)
            if request.status_code == 200:
                return json.loads(request.content).get('similarity')
        except:
            sleep(1)
            continue


def verif(pair_file):
    path_faceid_dic = {}
    for line in open('pic_path_faceid.txt'):
        tmp = line.rstrip().split()
        if len(tmp) == 2:
            pic_path = tmp[0]
            faceid = tmp[1]
            if faceid == 'no_face':
                continue
            path_faceid_dic[pic_path] = faceid

    f = open('pair_score.txt', 'w')
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            pic_path1 = tmp[0]
            pic_path2 = tmp[1]
            label = tmp[2]
            if pic_path1 in path_faceid_dic and pic_path2 in path_faceid_dic:
                similarity = verif_two_face(path_faceid_dic.get(pic_path1), path_faceid_dic.get(pic_path2))
                print '\t'.join(map(str, [pic_path1, pic_path2, label, similarity]))
                f.write('\t'.join(map(str, [pic_path1, pic_path2, label, similarity]))+'\n')


def merge():
    other_model_pair_score = {}
    face_pair_model = {}
    for line in open('pair_score.txt'):
        tmp = line.rstrip().split()
        if len(tmp) == 4:
            path1 = os.path.split(tmp[0])[1].replace('_face_0.jpg', '')
            path2 = os.path.split(tmp[1])[1].replace('_face_0.jpg', '')
            face_pair_model[(path1, path2)] = (tmp[2], tmp[3])
    f = open('merge.txt', 'w')
    f.write('\t'.join(map(str, ['path1', 'path2', 'label', 'face++', 'beiyou', 'deepface']))+'\n')
    for line in open('other_model_pair_score.txt'):
        if line.startswith('path1'):
            continue
        tmp = line.rstrip().split()
        if len(tmp) == 5:
            path1 = os.path.split(tmp[0])[1].replace('_face_0.jpg', '')
            path2 = os.path.split(tmp[1])[1].replace('_face_0.jpg', '')
            if (path1, path2) in face_pair_model:
                print face_pair_model.get((path1, path2))
                label, score = face_pair_model.get((path1, path2))
                face_pair_model[(path1, path2)] = (label, score, tmp[2], tmp[4])
                f.write('\t'.join(map(str, [path1, path2, label, score, tmp[2], tmp[4]]))+'\n')


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
                    similarity = verif_two_face(test_face_id, all_face_id)
                    # print similarity
                    this_similarity_list.append((similarity, all_pic_path))
                else:
                    # print 'all_tmp :', all_tmp
                    continue
        else:
            # print 'test_tmp :', test_tmp
            continue
        this_similarity_list.sort()
        print test_tmp[0], test_tmp[1], this_similarity_list[-1][0], this_similarity_list[-1][1]


if __name__ == '__main__':
    # create_face_id(pic_folder='C:\Users\liubo\Desktop\picture\skyeye_pic',
    #                result_file = 'C:\Users\liubo\Desktop\picture/skyeye_pic_face_id.txt')
    # create_face_id(pic_folder='C:\Users\liubo\Desktop\picture/face++_test_pic',
    #                result_file = 'C:\Users\liubo\Desktop\picture/face++_test_pic_face_id.txt')
    verif_all_person(all_person_file='C:\Users\liubo\Desktop\picture/skyeye_pic_face_id.txt',
                     test_person_file='C:\Users\liubo\Desktop\picture/face++_test_pic_face_id.txt')
    pass