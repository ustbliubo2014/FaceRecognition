# -*-coding:utf-8 -*-
__author__ = 'liubo-it'


import requests, json
import base64
import urllib2
import os
import pdb

storm_url= 'http://drpc1v.cldisk2.tjhc.qihoo.net:3775/drpc/hdp-iri_api'

print storm_url
def strom_run(post_data):
    print "online test"
    json_obj = json.loads(post_data)
    json_obj["debug"] = True
    post_data = json.dumps(json_obj)
    # print post_data
    requestPOST = urllib2.Request(data=post_data, url=storm_url)
    requestPOST.get_method = lambda: 'POST'
    f = urllib2.urlopen(requestPOST)
    recog_rlt = f.read()
    return recog_rlt

def face_cluster_run(post_data):
    return strom_run(post_data)

def test_image(image_name):
    # image_name = '00000040.png'
    image_file = open(image_name,'rb')
    encoded_string = base64.b64encode(image_file.read())
    request = {
            "version":"",
            "method": "get_face_info",
            "data":{
                "qid":"003",
                "imagedata":encoded_string,
                "image_id":image_name,
                "image_time":888

            }
        }
    tmp = face_cluster_run(json.dumps(request))
    # pdb.set_trace()
    print tmp


test_image('00000342.png')
# test_image('00000515.png')
test_image('00000600.png')
test_image('00000620.png')
# test_image('00000271.png')

