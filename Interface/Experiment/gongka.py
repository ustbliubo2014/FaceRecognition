# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: gongka.py
@time: 2017/1/4 17:55
@contact: ustb_liubo@qq.com
@annotation: gongka
"""
import sys
sys.path.append('/home/liubo-it/FaceRecognization/')
# from Interface.research_model import extract_feature_from_file
# from Interface.light_cnn_model import extract_feature_from_file
# from Interface.facenet_model import extract_feature_from_file
from sklearn.metrics.pairwise import cosine_similarity
import logging
from logging.config import fileConfig
import os
import msgpack_numpy
import base64
import pdb
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


def get_pic_feature(pic_folder, result_file):
    f_result = open(result_file, 'w')
    pic_list = os.listdir(pic_folder)
    for pic in pic_list:
        try:
            pic_path = os.path.join(pic_folder, pic)
            feature = extract_feature_from_file(pic_path)
            # print feature, feature.shape
            f_result.write(pic_path.decode('gbk')+'\t'+base64.b64encode(msgpack_numpy.dumps(feature))+'\n')
            print pic_path.decode('gbk')
        except:
            traceback.print_exc()
            continue
    f_result.close()


def verif_all_person(all_person_file, test_person_file):
    all_person_list = open(all_person_file).read().split('\n')
    test_person_list = open(test_person_file).read().split('\n')
    for test_element in test_person_list:
        try:
            test_tmp = test_element.split('\t')
            this_similarity_list = []
            if len(test_tmp) == 2:
                test_pic_path, test_face_feature = test_tmp[0], test_tmp[1]
                test_face_feature = msgpack_numpy.loads(base64.b64decode(test_face_feature))
                for all_element in all_person_list:
                    try:
                        all_tmp = all_element.split('\t')
                        if len(all_tmp) == 2:
                            all_pic_path, all_face_feature = all_tmp[0], all_tmp[1]
                            all_face_feature = msgpack_numpy.loads(base64.b64decode(all_face_feature))
                            similarity = cosine_similarity(all_face_feature, test_face_feature)[0][0]
                            # print similarity
                            this_similarity_list.append((similarity, all_pic_path))
                        else:
                            # print 'all_tmp :', all_tmp
                            continue
                    except:
                        traceback.print_exc()
                        continue
            else:
                # print 'test_tmp :', test_tmp
                continue
        except:
            traceback.print_exc()
            continue
        this_similarity_list.sort(key=lambda x:x[0])
        # print test_tmp[0], this_similarity_list[-1][0], this_similarity_list[-1][1], \
        #     this_similarity_list[-2][0], this_similarity_list[-2][1], \
        #     this_similarity_list[-3][0], this_similarity_list[-3][1], \
        #     this_similarity_list[-4][0], this_similarity_list[-4][1]
        print test_tmp[0], this_similarity_list[-1][1]


if __name__ == '__main__':
    test_pic_folder = '/tmp/test_pic'
    gongka_pic_folder = '/tmp/all_gongka_pic'
    test_pic_file = '/tmp/test_pic_face_id_facenet.txt'
    gongka_pic_file = '/tmp/all_gongka_pic_face_id_facenet.txt'

    # get_pic_feature(pic_folder=test_pic_folder, result_file=test_pic_file)
    # get_pic_feature(pic_folder=gongka_pic_folder, result_file=gongka_pic_file)
    verif_all_person(all_person_file=gongka_pic_file, test_person_file=test_pic_file)


