#-*- coding:utf-8 -*-
__author__ = 'liubo-it'

import json
import pdb

def analyse_image_result(json_data):
    dic = json.loads(json_data)
    data = dic.get('data')
    if data:
        data = data[0]
        image_id = data.get('image_id')
        face_id = data.get('face_id')
        user_id = data.get('user_id')
        return image_id, face_id, user_id
    else:
        return None, None, -1

# 将输出的结果按照cluster_id分别保存
def analyse_face_cluster_result(json_data):
    dic = json.loads(json_data)
    data = dic.get('data')
    clusters = data.get('cluster')
    print 'len(clusters)', len(clusters)
    for cluster in clusters:
        all_face = cluster.get('faces')
        for face in all_face:
            print face.get('image_id'), face.get('face_id'), face.get('prediction_cluster')

if __name__=='__main__':
    json_data = '{"msg": "ok", "data": [{"user_id": 0, "qid": "003", "image_time": 888, "detection": [2, 18, 117, 117], "image_id": "Adam_Driver_00000026.png", "face_id": "Adam_Driver_00000026.png_0"}], "func": "get_face_info", "rc": 0}'
    analyse_image_result(json_data)
