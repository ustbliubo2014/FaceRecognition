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
import time
import base64
import json
import sys
import os
import random
import json



#usr set

storm_url= 'http://drpc1v.cldisk2.tjhc.qihoo.net:3775/drpc/hdp-iri_api'

print storm_url
def strom_run(post_data):
    print "online test"
    json_obj = json.loads(post_data)
    json_obj["debug"] = True
    post_data = json.dumps(json_obj)
    print post_data
    requestPOST = urllib2.Request(
                data    =  post_data,
                url = storm_url
                        )
    requestPOST.get_method = lambda: 'POST'

    f = urllib2.urlopen(requestPOST)
    recog_rlt = f.read()

    return recog_rlt

def face_cluster_run(post_data):
    #return face_storm.run(post_data)
    return strom_run(post_data)

def write_txt_lines( txt_file_name , list_str_lines ,with_end = False ):
    if (not with_end ):
        list_str_lines = [line+'\n' for line in list_str_lines]

    file_object = open(txt_file_name, 'w')
    file_object.writelines(list_str_lines)
    file_object.close( )

def write_test_sh(txt_file_name , json_str):
    out_str = """storm_sync_send -idc bjdt hdp-iri_api '"""+json_str+"'"
    write_txt_lines(txt_file_name , [out_str])


def test_user():
    #a = input("ss")
    request = {
    "version":"",
    "method": "meta_add_user",
    "data":{
        "qid":"003",
    }
    }


    #print face_cluster_run(json.dumps(request))

    request = {
    "version":"",
    "method": "meta_add_user",
    "data":{
        "qid":"00d3",
    }
    }


    #print face_cluster_run(json.dumps(request))

    request = {
    "version":"",
    "method": "meta_get_user",
    "data":{
        "qid":"003",
    }
    }

    print request
    print face_cluster_run(json.dumps(request))


    request = {
    "version":"",
    "method": "meta_del_user",
    "data":{
        "qid":"003",
        "user_id":5
        }
    }


    print face_cluster_run(json.dumps(request))


    request = {
    "version":"",
    "method": "meta_get_user",
    "data":{
        "qid":"003",
    }
    }

    print face_cluster_run(json.dumps(request))

def test_image():

    Test_dir = '/home/lidongliang/Test/Demo/Train/'
    images = os.listdir(Test_dir)
    print images

    for each_image in images[4:5]:
        #a = input("ss")
        ##face_verfiy
        print "get_face_info"
        image_name = each_image
        with open(os.path.join(Test_dir,image_name), "rb") as image_file:
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
            print face_cluster_run(json.dumps(request))


#test tag face

# def test_tag():
    # qid = "003"

    # request = {
    # "version":"",
    # "method": "meta_add_label",
    # "data":{
        # "qid":qid,
        # "label":"ssss"
    # }
    # }


    # print face_cluster_run(json.dumps(request))

    # Test_dir = '/home/lidongliang/Test/Demo/Train/'

    # print "get_img_face_info"
    # image_name = "24979971-36020144175-3-1448934419888.jpg"
    # with open(os.path.join(Test_dir,image_name), "rb") as image_file:
        # encoded_string = base64.b64encode(image_file.read())


        # request = {
        # "version":"",
        # "method": "get_img_face_info",
        # "data":{
            # "qid":qid,
            # "imagedata":encoded_string,
            # "image_name":image_name,

        # }
        # }
        # print face_cluster_run(json.dumps(request))


    # request = {
    # "version":"",
    # "method": "tag_face",
        # "data":{
            # "qid":qid,
            # "label":"label11",
            # 'face_id':"24979971-36020144175-3-1448934419888.jpg_0"
        # }
    # }


    # print face_cluster_run(json.dumps(request))

    # print "get_img_face_info"
    # image_name = "24979971-36020144175-3-1448934419888.jpg"
    # with open(os.path.join(Test_dir,image_name), "rb") as image_file:
        # encoded_string = base64.b64encode(image_file.read())


        # request = {
        # "version":"",
        # "method": "get_img_face_info",
        # "data":{
            # "qid":qid,
            # "imagedata":encoded_string,
            # "image_name":image_name,

        # }
        # }
        # #print face_cluster_run(json.dumps(request))

    # request = {
    # "version":"",
    # "method": "stat_get_acc",
        # "data":{
            # "qid":qid,
        # }
    # }


    # print face_cluster_run(json.dumps(request))




def test_face_pre_cluster():
    print "face_pre_cluster"

    request = {
    "version":"",
    "method": "face_pre_cluster",
    "data":{
        "qids":["003"],
        "max_limit":40,

    }
    }

    print face_cluster_run(json.dumps(request))

def test_face_cluster():
    print "get_face_cluster"

    request = {
    "version":"",
    "method": "get_face_cluster",
    "data":{
        "qid":"003",
    }
    }

    print face_cluster_run(json.dumps(request))

def test_remove_face():
    qid = "003"

    print "remove_face"
    request = {
    "version":"",
    "method": "remove_face",
        "data":{
            "qid":qid,
            'face_ids':["24979971-36020144175-3-1448930773365.jpggVr_0"]
        }
    }


    print face_cluster_run(json.dumps(request))



def test_tag2():
    qid = "003"

    print "tag_face"
    request = {
    "version":"",
    "method": "tag_face",
        "data":{
            "qid":qid,
            "user_id":6,
            'face_ids':["24979971-36020144175-3-1448964609157.jpg_0"]
        }
    }


    print face_cluster_run(json.dumps(request))


def test_de_tag():
    qid = "003"

    print "de_tag_face"

    request = {
    "version":"",
    "method": "de_tag_face",
        "data":{
            "qid":qid,
            "user_id":5,
            'face_id':["24979971-36020144175-3-1448934581509.jpg_0"]
        }
    }


    print face_cluster_run(json.dumps(request))


def test_getFace():
    qid = "2520306912"

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

test_user()
#test_image()
#test_getFace()
#test_debug_set_cluster_threshold()
#test_face_pre_cluster()
#test_face_cluster()
#test_remove_face()
#test_face_cluster()
#test_tag2()
#test_de_tag()
for i in xrange(0):
    print i
    test_image()
for i in xrange(0):
    print i
    test_user()
    test_face_pre_cluster()
    test_face_cluster()
    test_tag2()
    test_de_tag()

#test_tag()
