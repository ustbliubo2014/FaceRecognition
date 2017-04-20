# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: recognition.py
@time: 2016/7/28 18:11
@contact: ustb_liubo@qq.com
@annotation: recognition
"""
import sys
import logging
from logging.config import fileConfig
import os
from util import load_all_path_label, port, local_request
import traceback
from sklearn.neighbors import LSHForest
import time
import numpy as np
from collections import Counter
import base64
import json
import sklearn.metrics.pairwise as pw
import cPickle
import cv2
import zlib
from scipy.misc import imread, imsave, imresize
import tornado
import msgpack

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


class FaceRecognition():
    def __init__(self, feature_dim, shape, verification_model_file):
        '''
            :param feature_dim: 模型的输出维度
            :param shape: 模型的图片尺寸
            :return:
        '''
        # 无法识别或程序出错时返回的str
        self.unknown = ''
        # 人脸数据库
        self.all_pic_data_folder = '/data/liubo/face/self'
        self.n_neighbors = 3
        self.lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=self.n_neighbors)
        self.all_labels = []
        self.all_pic_feature = []
        # 标识识别结果
        self.same_pic_id = 2
        self.must_be_same_id = 1
        self.must_be_not_same_id = 0
        self.maybe_same_id = 3
        # 相关字符串
        self.new_person_str = 'new_person_'
        self.must_same_str = '_Must_Same'
        self.maybe_same_str = '_Maybe_same'
        self.current_new_person_id = self.find_current_new_person_id()
        self.load_time = time.time()
        self.user_count = {}
        self.feature_dim = feature_dim
        self.shape = shape
        self.verification_model_file = verification_model_file
        self.verification_model = cPickle.load(open(self.verification_model_file, 'rb'))
        self.verification_same_person = 0     # 0代表是一个人

    def find_current_new_person_id(self):
        # 当前new_person_*中最大的id+1
        max_old_person_id = 0
        person_list = os.listdir(self.all_pic_data_folder)
        for person in person_list:
            try:
                if person.startswith(self.new_person_str):
                    tmp = person[len(self.new_person_str):].split('_')
                    if len(tmp) > 0 and max_old_person_id < int(tmp[0]):
                        max_old_person_id = int(tmp[0])
            except:
                traceback.print_exc()
                continue
        current_new_person_id = max_old_person_id + 1
        return current_new_person_id

    def extract_feature_from_numpy(self, pic_data):
        # 在子类中实现
        pass

    def extract_pic_feature(self, pic_data):
        pic_feature = np.zeros(shape=(pic_data.shape[0], self.feature_dim))
        for index in range(pic_data.shape[0]):
            pic_feature[index: (index+1)] = \
                self.extract_feature_from_numpy(pic_data[index: (index+1)])
        return pic_feature

    def read_one_rgb_pic(self, pic_path):
        # 在子类中实现
        pass

    def load_all_pic_data(self, all_pic_path):
        all_pic_data = []
        for pic_path in all_pic_path:
            all_pic_data.append(self.read_one_rgb_pic(pic_path))
        all_pic_data = np.asarray(all_pic_data)
        return all_pic_data

    def load_all_data(self):
        # 将以前标记的数据全部读入,用LSH Forest保存,方便计算距离
        all_pic_path, all_label = load_all_path_label(self.all_pic_data_folder)
        if len(all_label) == 0:
            return
        all_pic_data = self.load_all_pic_data(all_pic_path)
        all_pic_feature = self.extract_pic_feature(all_pic_data)
        self.lshf.fit(all_pic_feature, all_label)
        self.all_pic_feature = list(all_pic_feature)
        self.all_labels = list(all_label)
        self.user_count = Counter(self.all_labels)
        print self.user_count

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
                    face = self.read_one_rgb_pic(pic_path)
                    add_flag = local_request(face, person)
                    if add_flag:
                        add_num += 1
                    else:
                        print 'not add :', pic_path
        end = time.time()
        if add_num > 0:
            self.load_time = end
            print 'add pic num :', add_num, 'Dynamic increase time :', (end - start)


    def add_one_new_pic(self, img_str, label):
        # 这时读入的数据
        try:
            # 读入数据时已经转换成需要的尺寸
            image = base64.decodestring(img_str)
            im = np.fromstring(image)
            im_feature = self.extract_feature_from_numpy(im)
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
        try:
            self.lshf.partial_fit(one_pic_feature.reshape(1, self.feature_dim), pic_label)
            self.all_labels.append(pic_label)
            self.all_pic_feature.append(np.reshape(one_pic_feature, newshape=(1, one_pic_feature.size)))
        except:
            traceback.print_exc()
            return


    def find_k_neighbors_with_lsh(self, one_pic_feature):
        '''
            :return [(dist, verificate_result, label)]
        '''
        try:
            tmp = self.lshf.kneighbors(one_pic_feature.reshape(1, self.feature_dim),
                                       n_neighbors=self.n_neighbors, return_distance=True)
            neighbors_label = np.asarray(self.all_labels)[tmp[1][0]]
            neighbors_feature = np.asarray(self.all_pic_feature)[tmp[1][0]]
            verificate_result_list = []
            for index in range(len(neighbors_feature)):
                pair_score = pw.cosine_similarity(neighbors_feature[index].reshape(1, self.feature_dim),
                                     one_pic_feature.reshape(1, self.feature_dim))[0][0]
                verificate_result_list.append(self.verification_model.predict(pair_score))
            return zip(tmp[0][0], verificate_result_list, neighbors_label)
        except:
            return None

    def evaluate_result(self, dist_label_list):
        '''
            :param result: [(dist, verificate_result, label)]
            :return: this_id(Must_same, Must_not_same, May_same), this_label(人名)
        '''
        if dist_label_list == None:
            return self.must_be_not_same_id, self.new_person_str + str(self.current_new_person_id)
        else:
            try:
                for element in dist_label_list:
                    try:
                        this_dist, this_same_person_result, this_label = element
                        if this_same_person_result == self.verification_same_person:
                            return self.must_be_same_id, this_label
                    except:
                        continue
            except:
                return self.must_be_not_same_id, ''
        # 都没有找到一样的, 返回一个新人的id
        return self.must_be_not_same_id, self.new_person_str + str(self.current_new_person_id)

    def recvive_img(self, img):
        # 将接收到的str中转换成numpy [不同模型可能不一样]
        pass

    def save_pic(self, pic_array, pic_path):
        # [不同模型的数据维度不同, 子类实现]
        pass

    def online_cluster(self, this_id, this_label):
        '''
            :param this_id: evaluate后的结果
            :param this_label: 人名
            :return: need_save(是否保存图片), need_add(图片是否添加到内存)
        '''
        need_add = False
        need_save = False
        this_person_folder = ''
        try:
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
        except:
            need_add = False
            need_save = False
            this_person_folder = ''
            return need_add, need_save, this_person_folder
        return need_add, need_save, this_person_folder

    def recognize_online_cluster(self, image, image_id):
        '''
            将得到的图片进行识别, 不是库里的人则认为是一个新人
            生成一个新的id并将不再库里的新人加入LSH Forest
            :return: 名字, 可信度, '', '' (后两个参数是为了兼容以前程序)
        '''
        start = time.time()
        try:
            image = zlib.decompress(base64.decodestring(image))
            im = self.recvive_img(image)
        except:
            traceback.print_exc()
            return self.unknown, 1.0, '', ''
        try:
            # 流程 : 找距离最近的图片 ; 计算prob ; 在线聚类 ; 加入LSH Forest
            im_feature = self.extract_feature_from_numpy(im)
            # 找距离最近的图片 --- 用LSH Forest 找出最近的10张图片,然后分别计算距离
            dist_label_list = self.find_k_neighbors_with_lsh(im_feature)
            logger_error.error(str(dist_label_list))
            # 计算
            this_id, this_label = self.evaluate_result(dist_label_list)
            # 在线聚类 --- 根据dist确定是重新增加一个人还是加入到已有的人中
            need_add, need_save, this_person_folder = self.online_cluster(this_id, this_label)
            if need_save:
                try:
                    if not os.path.exists(this_person_folder):
                        os.makedirs(this_person_folder)
                    this_pic_name = os.path.join(this_person_folder, image_id+'.png')
                    self.save_pic(im, this_pic_name)
                except:
                    traceback.print_exc()
                    pass
            # 加入LSH Forest --- partial_fit
            if need_add:
                self.add_one_pic(im_feature, this_label)
                # 根据label和image_id可以存生成文件名,确定是否要存储文件[可以选择在服务器和本地同时存储]
            if this_id == self.same_pic_id or this_id == self.must_be_not_same_id or this_id == self.must_be_same_id:
                end = time.time()
                print 'recog time :',(end - start)
                return this_label.replace(self.must_same_str, ''), str(dist_label_list[0][0]), '', ''
            else:
                # 灰度区域,不显示人名
                return self.unknown, str(dist_label_list[0][0]), '', ''
        except:
            traceback.print_exc()
            return self.unknown, str(100.0), '', ''



class MainHandler(tornado.web.RequestHandler):
    def post(self):
        request_type = self.get_body_argument('request_type')
        if request_type == 'recognization':
            try:
                image_id = self.get_body_argument("image_id")
                image = self.get_body_argument("image")
                print "receive image", image_id, time.time()
                result = face_recognition.recognize_online_cluster(image, image_id)
                result = base64.b64encode(msgpack.dumps(result))
                self.write(json.dumps({"recognization": result}))
            except:
                traceback.print_exc()
                return
        elif request_type == 'add': # 向LSH Forest中加入新图片
            one_pic_str = self.get_body_argument("one_pic_feature")
            label = self.get_body_argument("label")
            self.write(json.dumps({"add": face_recognition.add_one_new_pic(one_pic_str, label)}))


def server(application):
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()


def add_new_pic(face_recognization):
    while True:
        # 直接请求本地的服务,加入LSH Forest
        face_recognization.add_all_new_pic()
        time.sleep(10)


if __name__ == "__main__":
    sub_process_id = ''
    try:
        face_recognition = FaceRecognition()
        face_recognition.load_all_data()
        application = tornado.web.Application([(r"/", MainHandler),])

        add_new_pic_args = (face_recognition, )
        add_new_pic_thread = MyThread(func=add_new_pic, args=add_new_pic_args, name='add_new_pic')

        add_new_pic_thread.start()

        application.listen(port)
        tornado.ioloop.IOLoop.instance().start()

        add_new_pic_thread.join()
    except:
        traceback.print_exc()


