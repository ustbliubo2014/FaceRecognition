# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: util.py
@time: 2016/7/28 18:12
@contact: ustb_liubo@qq.com
@annotation: util
"""
import sys
import logging
from logging.config import fileConfig
import os
import base64
import urllib2
import urllib
import requests
import traceback
import json

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

port = 6666


def load_all_path_label(data_folder):
    '''
        不同的模型有不同的读数据的办法(avg可能不同, 维度转换方式也可能不同[RBG->BRG])
        :param data_folder: 包含所有人的文件夹
        :return:
    '''
    all_pic_path = []
    all_label = []
    person_list = os.listdir(data_folder)
    for person in person_list:
        if person == 'unknown' or 'Must_Same' in person or 'Maybe_same' in person:
            continue
        person_path = os.path.join(data_folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            all_pic_path.append(os.path.join(person_path, pic))
    return all_pic_path, all_label


def local_request(face, person):
    # 请求本地服务
    try:
        request = {"label": person, "request_type": 'add',
                "one_pic_feature": base64.encodestring(face.tostring())}
        response = requests.post(url="http://127.0.0.1:%d/" % port, data=request)
        add_flag = json.loads(response.content)['add']
        return add_flag
    except:
        traceback.print_exc()
        return None
    # # pdb.set_trace()
    # requestPOST = urllib2.Request(
    #                     data=urllib.urlencode(request),
    #                     url="http://127.0.0.1:%d/"%port
    # )
    # requestPOST.get_method = lambda : "POST"
    # try:
    #     s = urllib2.urlopen(requestPOST).read()
    # except urllib2.HTTPError, e:
    #     print e.code
    # except urllib2.URLError, e:
    #     print str(e)
    # try:
    #     add_flag = json.loads(s)["add"]
    #     if not add_flag:# 加载失败
    #         print 'no add file :', pic_path
    #     else:
    #         add_num += 1
    # except:
    #     print 'no add file :', pic_path
    #     traceback.print_exc()


if __name__ == '__main__':
    pass
