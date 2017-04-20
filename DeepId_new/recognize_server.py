# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: recognize_server.py
@time: 2016/7/4 17:18
@contact: ustb_liubo@qq.com
@annotation: recognize_server: 提供线上服务
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='recognize_server.log',
                    filemode='w')


import numpy as np
import time
import json
import base64
import tornado.ioloop
import tornado.web
import cPickle
import traceback
from scipy.misc import imresize, imsave, imread
import os
import msgpack
from collections import Counter
from sklearn.neighbors import LSHForest
from util.load_data import load_rgb_multi_person_all_data
from MyThread import MyThread
import urllib2
import urllib
import zlib
from DeepId_batch_model import load_deepid_model


class FaceRecognization():
    def __init__(self):
        self.deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
        self.deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'
        print 'load deepid model'
        self.model, self.get_Conv_FeatureMap = load_deepid_model(self.deepid_model_file, self.deepid_weight_file)
        self.unknown = ''
        self.same_person_num = 1
        self.has_save_pic_feature = []
        self.has_cal_dist = []
        self.pic_shape = (50, 50, 3)
        self.NeighbourNum = 1
        # self.all_pic_data_folder = '/data/liubo/face/self_train'
        self.all_pic_data_folder = '/data/liubo/face/self'
        # self.other_dataset_para_add = 0.5
        # self.all_pic_data_folder = '/data/liubo/face/vgg_face_dataset/train'
        self.other_dataset_para_add = 1
        self.n_neighbors = 10
        self.lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=self.n_neighbors)
        self.all_labels = []
        self.all_pic_feature = []
        # 直接使用LSH Forest的distance
        self.same_pic_distance = 10 #可能是同一张图片
        self.must_be_same_threshold = 11 + self.other_dataset_para_add # 是同一个人
        self.almost_be_same_threshold = 12.5 + self.other_dataset_para_add # 最近的几个dist都小于12且label相同
        self.almost_num = 3
        self.must_not_be_same_threshold = 14 + self.other_dataset_para_add # 不是同一个人
        self.same_pic_id = 2
        self.must_be_same_id = 1
        self.must_be_not_same_id = 0
        self.maybe_same_id = 3
        self.prob_model_file = '/data/liubo/face/vgg_face_dataset/model/dist_prob.p' # 根据距离判断是同一个人的概率
        self.prob_model = cPickle.load(open(self.prob_model_file, 'rb'))
        self.new_person_str = 'new_person_'
        self.current_new_person_id = self.find_current_new_person_id()
        self.must_same_str = '_Must_Same'
        self.maybe_same_str = '_Maybe_same'
        self.filter_list = [self.new_person_str, self.must_same_str, self.maybe_same_str]
        self.load_time = time.time()
        self.user_count = {}


    def find_current_new_person_id(self):
        old_person_id = []
        person_list = os.listdir(self.all_pic_data_folder)
        for person in person_list:
            if person.startswith(self.new_person_str):
                tmp = person[len(self.new_person_str):].split('_')
                if len(tmp) > 0:
                    this_id = int(tmp[0])
                    old_person_id.append(this_id)
        if len(old_person_id) == 0:
            current_new_person_id = 0
        else:
            current_new_person_id = max(old_person_id) + 1
        print 'current_new_person_id :', current_new_person_id
        return current_new_person_id


    def extract_pic_feature(self, pic_data, batch_size=128, feature_dim=1024):
        pic_feature = np.zeros(shape=(pic_data.shape[0], feature_dim))
        batch_num = pic_data.shape[0] / batch_size
        for index in range(batch_num):
            pic_feature[index*batch_size:(index+1)*batch_size, :] = \
                self.get_Conv_FeatureMap([pic_data[index*batch_size:(index+1)*batch_size],0])[0]
        if batch_num*batch_size < pic_data.shape[0]:
            pic_feature[batch_num*batch_size:, :] = self.get_Conv_FeatureMap([pic_data[batch_num*batch_size:],0])[0]
        return pic_feature


    def load_all_data(self):
        # 将以前标记的数据全部读入,用LSH Forest保存,方便计算距离
        train_data, train_label = load_rgb_multi_person_all_data(all_person_folder=self.all_pic_data_folder,
                                pic_shape=self.pic_shape, label_int=False, person_num_threshold=None,
                                pic_num_threshold=None, filter_list=self.filter_list,func=None)
        if len(train_label) == 0:
            return
        pic_feature = self.extract_pic_feature(train_data)
        start = time.time()
        self.lshf.fit(pic_feature, train_label)
        self.all_pic_feature = list(pic_feature)
        self.all_labels = list(train_label)
        end = time.time()
        self.load_time = end
        self.user_count = Counter(self.all_labels)
        print self.user_count, 'fit all data time :', (end - start)


    def add_all_new_pic(self):
        '''
            将从上次加载数据到当前新增的文件都加载到LSH Forest(有可能是新增加一个人,还有可能是对已有的人增加新图片)
            遍历文件夹(self.all_pic_data_folder),根据文件的时间判断是否需要加入该图片
        '''
        start = time.time()
        person_list = os.listdir(self.all_pic_data_folder)
        add_num = 0
        for person in person_list:
            if self.must_same_str in person or self.maybe_same_str in person or self.new_person_str in person:
                continue
            person_path = os.path.join(self.all_pic_data_folder, person)
            if not os.path.isdir(person_path):
                continue
            pic_list = os.listdir(person_path)
            for pic in pic_list:
                pic_path = os.path.join(person_path, pic)
                last_modify_time = os.stat(pic_path).st_atime
                if last_modify_time > self.load_time:
                    face = imresize(imread(pic_path), self.pic_shape)
                    # print 'face.shape :', face.shape
                    # face = np.transpose(np.reshape(face,newshape=(1, 50, 50, 3)), (0,3,1,2))
                    # 请求本地服务
                    request = {
                        "label": person,
                        "request_type": 'add',
                        "one_pic_feature": base64.encodestring(face.tostring())
                    }
                    # pdb.set_trace()
                    requestPOST = urllib2.Request(
                        data=urllib.urlencode(request),
                        url="http://127.0.0.1:6666/"
                    )
                    requestPOST.get_method = lambda : "POST"
                    try:
                        s = urllib2.urlopen(requestPOST).read()
                    except urllib2.HTTPError, e:
                        print e.code
                    except urllib2.URLError, e:
                        print str(e)
                    try:
                        add_flag = json.loads(s)["add"]
                        if not add_flag:# 加载失败
                            print 'no add file :', pic_path
                        else:
                            add_num += 1
                    except:
                        print 'no add file :', pic_path
                        traceback.print_exc()
                        continue
                    add_num += 1
        end = time.time()
        if add_num > 0:
            self.load_time = end
            print 'self.load_time', self.load_time
            print 'add pic num :', add_num, 'Dynamic increase time :', (end - start)


    def add_one_new_pic(self, img_str, label):
        try:
            image = base64.decodestring(img_str)
            im = np.fromstring(image, dtype=np.uint8)
            # pdb.set_trace()
            im = np.reshape(im, newshape=(1, 50, 50, 3))
            im = np.asarray(im, dtype=np.float32) / 255.0
            im = np.transpose(im,(0,3,1,2))
            im_feature = self.get_Conv_FeatureMap([im,0])[0]
            self.add_one_pic(im_feature, label)
            return True
        except:
            traceback.print_exc()
            return False


    def add_one_pic(self, one_pic_feature, pic_label):
        '''
            将一个图像的特征加入到LSH Forest,同时将对应的标签加入到self.all_labels
            :param pic_feature: array shape :(1,1024)
            :param pic_label: (1,)
            :return:
        '''
        self.lshf.partial_fit(one_pic_feature, pic_label)
        self.all_labels.append(pic_label)
        self.all_pic_feature.append(np.reshape(one_pic_feature,newshape=(one_pic_feature.size)))


    def find_k_neighbors_with_lsh(self, one_pic_feature):
        try:
            tmp = self.lshf.kneighbors(one_pic_feature, n_neighbors=self.n_neighbors, return_distance=True)
            result_label = np.asarray(self.all_labels)[tmp[1][0]]
            return zip(tmp[0][0], result_label)
        except:
            return None


    def cal_proba(self, dist):
        # 根据dist计算准确率(基于以前的统计结果)
        return self.prob_model.predict_proba(dist)


    def evaluate_result(self, result):
        # 根据计算的距离和label的情况确定结果(属于哪一个人,概率,是否存储等情况); result已根据dist排序
        # 最小dist小于10;前几张图片的dist都小于13且label相同;
        # 如果同时存在new_person和name,先处理name
        name_result = [element for element in result if self.new_person_str not in element[1]]
        if len(name_result) > 0:
            if name_result[0][0] < self.same_pic_distance:
                return self.same_pic_id, name_result[0][1]
            elif name_result[0][0] <= self.must_be_same_threshold:
                return self.must_be_same_id, name_result[0][1]

        if result[0][0] <= self.same_pic_distance:
            return self.same_pic_id, result[0][1] # 已含有这张图片(不用加入LSH Forest), label
        elif result[0][0] <= self.must_be_same_threshold:
            return self.must_be_same_id, result[0][1] # 可能确定为一个人(需要加入LSH Forest), label
        else:
            if result[0][0] >= self.must_not_be_same_threshold:
                # 标记为一个新的人
                return self.must_be_not_same_id, ''
            else:# 灰度区域 --- 不显示人名
                if result[0][0] <= self.almost_be_same_threshold:
                    almost_label = result[0][0]
                    for index in range(1, self.almost_num):
                        if result[index][0] < self.almost_be_same_threshold and result[index][1] == almost_label:
                            continue
                        else:
                            # 可能是数据库里的一个人
                            return self.maybe_same_id, result[0][1]
                    # 是数据库里的一个人
                    return self.must_be_same_id, result[0][1] # 可能确定为一个人(需要加入LSH Forest), label
                else:
                    # 可能是数据库里的一个人
                    return self.maybe_same_id, result[0][1]


    def recognize_online_cluster(self, image, image_id):
        '''
            :param image: 将得到的图片进行识别,加入的LSH Forest,根据距离计算proba(不同的距离对应不同的准确率,根据已有的dist计算阈值);
                            和已经设定的阈值判断是不是一个新出现的人,确定是原来已有的人,还是不确定是原来已有的人
            :return:
        '''
        start = time.time()
        need_add = False
        has_save_num = 0
        try:
            # image = base64.decodestring(image)
            image = base64.decodestring(image)
            image = zlib.decompress(image)
            im = np.fromstring(image, dtype=np.uint8)
            im = np.reshape(im, newshape=(1, 50, 50, 3))
            im_raw = im[0][:, :, :]
            im = np.asarray(im,dtype=np.float32) / 255.0
            im = np.transpose(im,(0,3,1,2))
            end_load = time.time()
            # print 'load time :', (end_load -start)
        except:
            traceback.print_exc()
            return self.unknown, 1.0, self.has_save_pic_feature, need_add
        try:
            # 流程 : 找距离最近的图片 ; 计算prob ; 在线聚类 ; 加入LSH Forest
            im_feature = self.get_Conv_FeatureMap([im,0])[0]

            # 找距离最近的图片 --- 用LSH Forest 找出最近的10张图片,然后分别计算距离
            dist_label_list = self.find_k_neighbors_with_lsh(im_feature)
            print dist_label_list
            if dist_label_list == None:
                this_id = self.must_be_not_same_id
                this_label = self.new_person_str + str(self.current_new_person_id)
            else:
                # 计算prob --- 根据距离计算prob
                this_id, this_label = self.evaluate_result(dist_label_list)
            # print 'this_id :', this_id, 'this_label :', this_label
            # 在线聚类 --- 根据dist确定是重新增加一个人还是加入到已有的人中
            need_save = False
            if this_id == self.same_pic_id:
                need_add = False
            elif this_id == self.must_be_same_id:
                need_add = False
                need_save = True
                this_person_folder = os.path.join(self.all_pic_data_folder, this_label+self.must_same_str)
            elif this_id == self.must_be_not_same_id:
                this_label = self.new_person_str + str(self.current_new_person_id)
                self.current_new_person_id += 1
                this_person_folder = os.path.join(self.all_pic_data_folder, this_label)
                need_add = True
                need_save = True
            elif this_id == self.maybe_same_id:
                this_person_folder = os.path.join(self.all_pic_data_folder, this_label+self.maybe_same_str)
                need_add = False # prob在灰度区域的不如入,其余情况加入
                need_save = True
            else:
                print 'error para :', this_id
            if need_save:
                try:
                    if not os.path.exists(this_person_folder):
                        os.makedirs(this_person_folder)
                    this_pic_name = os.path.join(this_person_folder, image_id+'.png')
                    # print 'this_pic_name :', this_pic_name
                    imsave(this_pic_name, im_raw)
                except:
                    traceback.print_exc()
                    return self.unknown, 1.0, has_save_num, False

            # 加入LSH Forest --- partial_fit
            if need_add:
                self.add_one_pic(im_feature, this_label)
                has_save_num += 1
                # 根据label和image_id可以存生成文件名,确定是否要存储文件[可以选择在服务器和本地同时存储]
            if this_id == self.same_pic_id or this_id == self.must_be_not_same_id or this_id == self.must_be_same_id:
                end = time.time()
                print 'recog time :',(end - start)
                return this_label.replace(self.must_same_str, ''), str(dist_label_list[0][0]), str(has_save_num), str(need_add)
            else:
                # 灰度区域,不显示人名
                return self.unknown, str(dist_label_list[0][0]), str(has_save_num), str(need_add)
        except:
            traceback.print_exc()
            return self.unknown, str(100.0), str(has_save_num), str(False)



class MainHandler(tornado.web.RequestHandler):
    def post(self):
        request_type = self.get_body_argument('request_type')
        if request_type == 'recognization':
            try:
                image_id = self.get_body_argument("image_id")
                image = self.get_body_argument("image")
                print "receive image", image_id, time.time()
                result = face_recognization.recognize_online_cluster(image, image_id)
                result = base64.b64encode(msgpack.dumps(result))
                self.write(json.dumps({"recognization": result}))
            except:
                traceback.print_exc()
                return
        elif request_type == 'add': # 向LSH Forest中加入新图片
            one_pic_str = self.get_body_argument("one_pic_feature")
            label = self.get_body_argument("label")
            self.write(json.dumps({"add": face_recognization.add_one_new_pic(one_pic_str, label)}))



def server(application):
    application.listen(6666)
    tornado.ioloop.IOLoop.instance().start()



def add_new_pic(face_recognization):
    while True:
        # 直接请求本地的服务,加入LSH Forest
        face_recognization.add_all_new_pic()
        time.sleep(10)



if __name__ == "__main__":
    sub_process_id = ''
    try:
        face_recognization = FaceRecognization()
        face_recognization.load_all_data()
        application = tornado.web.Application([(r"/", MainHandler),])

        add_new_pic_args = (face_recognization, )
        add_new_pic_thread = MyThread(func=add_new_pic, args=add_new_pic_args, name='add_new_pic')

        add_new_pic_thread.start()

        application.listen(6666)
        tornado.ioloop.IOLoop.instance().start()

        add_new_pic_thread.join()
    except:
        traceback.print_exc()

