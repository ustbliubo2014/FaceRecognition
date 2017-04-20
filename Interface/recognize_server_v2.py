# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: recognize_server_v2.py
@time: 2016/11/4 18:12
@contact: ustb_liubo@qq.com
@annotation: recognize_server_v2 : 重构, 方便切换模型
"""
import sys
sys.path.insert(0, '/home/liubo-it/FaceRecognization/')
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
import shutil
from collections import Counter
import msgpack_numpy
import pdb
from sklearn.neighbors import LSHForest
from MyThread import MyThread
import zlib
import logging
from logging.config import fileConfig
import sklearn.metrics.pairwise as pw
import cv2
import stat
import math
from DetectAndAlign.align_interface import align_face
from recog_util import read_one_rgb_pic, is_blur, avg, image_request, get_time_slot, get_current_day
from collections import deque


# 研究院的接口和现在的接口不方便放在一起, 两个分开写
# (可以用来替换自己的模型)
if len(sys.argv) == 1 or sys.argv[1] == 'vgg':
    model_label = 'vgg'
    # 使用vgg_model
    from vgg_model import (PIC_SHAPE, pic_shape, upper_verif_threshold, lower_verif_threshold, FEATURE_DIM,
                        extract_feature_from_file, extract_feature_from_numpy, verification_model,
                        verification_same_person, port, nearest_num, nearest_time_threshold)
elif sys.argv[1] == 'research':
    model_label = 'research'
    from research_model import (extract_feature_from_file, lower_verif_threshold, PIC_SHAPE, pic_shape,
                        upper_verif_threshold, FEATURE_DIM, verification_model, extract_feature_from_numpy,
                        verification_same_person, port, nearest_num, nearest_time_threshold)
elif sys.argv[1] == 'light_cnn':
    model_label = 'light_cnn'
    from light_cnn_model import (PIC_SHAPE, pic_shape, upper_verif_threshold, lower_verif_threshold, FEATURE_DIM,
        extract_feature_from_file, extract_feature_from_numpy, verification_model, verification_same_person, port,
        nearest_num, same_pic_threshold)
else:
    print 'error model para'
    sys.exit()

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


check_port = 7777
check_ip = '10.160.164.26'
tmp_face_dir = '/tmp/face_recog_tmp'
log_dir = model_label+'_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def load_train_data(data_folder):
    all_pic_data = []
    all_label = []
    person_list = os.listdir(data_folder)
    for person in person_list:
        if person == 'unknown' or 'Must_Same' in person or 'Maybe_same' in person:
            continue
        person_path = os.path.join(data_folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            pic_array = read_one_rgb_pic(pic_path, pic_shape)
            all_pic_data.append(pic_array[0])
            all_label.append(person)
    all_pic_data = np.asarray(all_pic_data)
    all_label = np.asarray(all_label)
    return all_pic_data, all_label


class FaceRecognition():
    def __init__(self):
        self.unknown = ''
        self.same_person_num = 1
        self.has_save_pic_feature = []
        self.has_cal_dist = []
        self.NeighbourNum = 10
        self.all_pic_data_folder = '/data/liubo/face/self'
        self.other_dataset_para_add = 1
        self.n_neighbors = 5
        self.lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=self.n_neighbors)
        self.all_labels = []
        self.all_pic_feature = []
        self.same_pic_id = 2
        self.must_be_same_id = 1
        self.must_be_not_same_id = 0
        self.maybe_same_id = 3
        self.new_person_str = 'new_person_'
        self.current_new_person_id = self.find_current_new_person_id()
        self.must_same_str = '_Must_Same'
        self.maybe_same_str = '_Maybe_same'
        self.load_time = time.time()
        self.user_count = {}
        # 不同的模型阈值不相同
        self.upper_threshold = upper_verif_threshold
        self.lower_threshold = lower_verif_threshold
        self.same_pic_threshold = same_pic_threshold
        self.pitch_threshold = 20
        self.yaw_threshold = 20
        self.roll_threshold = 20
        #  [(time, feature),...,(time, feature)] : 根据时间计算当前图片与前5张图片的相似度(如果时间相差很多, 不在计算)
        self.nearest = deque(maxlen=nearest_num)
        self.trans_dic = {self.same_pic_id: 'same_pic', self.must_be_same_id: 'must_same_id',
                          self.must_be_not_same_id: 'must_not_same_id', self.maybe_same_id: 'maybe_same_id'}
        self.verification_same_person = 0


    def cal_nearest_sim(self, current_feature):
        nearest_sim_list = []
        try:
            length = len(self.nearest)
            for k in range(length):
                try:
                    person_name, pre_feature = self.nearest[k]
                    # 不在考虑时间, 只考虑图片的相似度
                    this_sim = pw.cosine_similarity(np.reshape(np.asarray(pre_feature), (1, len(pre_feature))),
                                                    np.reshape(np.asarray(current_feature), (1, len(current_feature))))

                    nearest_sim_list.append((this_sim, verification_model.predict(this_sim), person_name))
                except:
                    traceback.print_exc()
                    continue
            return nearest_sim_list
        except:
            traceback.print_exc()
            return nearest_sim_list


    def find_current_new_person_id(self):
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')

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
        log_file.write('\t'.join(map(str, ['current_new_person_id :', current_new_person_id]))+'\n')
        log_file.close()
        return current_new_person_id


    def extract_pic_feature(self, pic_data, batch_size=1, feature_dim=FEATURE_DIM):
        '''
            用于提取多张图片的特征(用于处理load数据)
            :param pic_data: 图片数据
            :param batch_size:
            :param feature_dim: 模型输出维度(vgg的输出是4096)
            :return:
        '''
        pic_feature = np.zeros(shape=(pic_data.shape[0], feature_dim))
        batch_num = pic_data.shape[0] / batch_size
        for index in range(batch_num):
            pic_feature[index*batch_size:(index+1)*batch_size, :] = \
                extract_feature_from_numpy(pic_data[index*batch_size:(index+1)*batch_size])
        if batch_num*batch_size < pic_data.shape[0]:
            pic_feature[batch_num*batch_size:, :] = \
                extract_feature_from_numpy(pic_data[batch_num*batch_size:])
        return pic_feature


    def load_all_data(self):
        # 将以前标记的数据全部读入,用LSH Forest保存,方便计算距离
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')

        train_data, train_label = load_train_data(self.all_pic_data_folder)
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
        log_file.write('\t'.join(map(str, [self.user_count,
                                           'fit all data time :', (end - start)]))+'\n')
        log_file.close()


    def add_all_new_pic(self):
        '''
            将从上次加载数据到当前新增的文件都加载到LSH Forest(有可能是新增加一个人,还有可能是对已有的人增加新图片)
            遍历文件夹(self.all_pic_data_folder),根据文件的时间判断是否需要加入该图片
            用户新加入的图片先进行人脸检测, 如果能够检测到人脸，使用检测结果， 否则使用用户的原始图片
        '''
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')

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
                    # 请求本地服务
                    request = {
                        "label": person,
                        "request_type": 'add',
                        "one_pic_feature": pic_path
                    }
                    url = "http://127.0.0.1:%d/"%port
                    result = image_request(request, url)
                    try:
                        add_flag = json.loads(result)["add"]
                        if not add_flag:# 加载失败
                            log_file.write('\t'.join(map(str, ['no add file :', pic_path]))+'\n')
                        else:
                            add_num += 1
                    except:
                        log_file.write('\t'.join(map(str, ['no add file :', pic_path]))+'\n')
                        traceback.print_exc()
                        continue
                    add_num += 1
        end = time.time()
        if add_num > 0:
            self.load_time = end
            log_file.write('\t'.join(map(str, ['self.load_time', self.load_time]))+'\n')
            log_file.write('\t'.join(map(str, ['add pic num :', add_num,
                                               'Dynamic increase time :', (end - start)]))+'\n')
            log_file.close()
        else:
            log_file.close()


    def add_one_new_pic(self, pic_path, label):
        try:
            # 读入数据时已经转换成需要的尺寸
            im_feature = extract_feature_from_file(pic_path)
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
        self.lshf.partial_fit(one_pic_feature.reshape(1, FEATURE_DIM), pic_label)
        self.all_labels.append(pic_label)
        self.all_pic_feature.append(np.reshape(one_pic_feature, newshape=(1, one_pic_feature.size)))


    def find_k_neighbors_with_lsh(self, one_pic_feature):
        '''
            :param one_pic_feature: 图像特征
            :return: 需要返回neighbors的特征, 用于计算pariwise
        '''
        try:
            tmp = self.lshf.kneighbors(one_pic_feature.reshape(1, FEATURE_DIM), n_neighbors=self.n_neighbors, return_distance=True)
            neighbors_label = np.asarray(self.all_labels)[tmp[1][0]]
            neighbors_feature = np.asarray(self.all_pic_feature)[tmp[1][0]]
            pair_score_list = []
            cos_sim_list = []
            for index in range(len(neighbors_feature)):
                pair_score = pw.cosine_similarity(neighbors_feature[index].reshape(1, FEATURE_DIM),
                                     one_pic_feature.reshape(1, FEATURE_DIM))[0][0]
                cos_sim_list.append(pair_score)
                pair_score_list.append(verification_model.predict(pair_score))
            result = zip(cos_sim_list, pair_score_list, neighbors_label)
            # result = self.filter_result(result)
            # result.sort(key=lambda x:x[0], reverse=True)
            return result
        except:
            return None


    def filter_result(self, result):
        '''
            :param result: [(cos_sim, same_person_result, label),
                            (cos_sim, same_person_result, label),
                            (cos_sim, same_person_result, label)] 按cos_sim降序排列
            :return: this_id(Must_same, Must_not_same, May_same), this_label(人名)
        '''
        # 分值相同的, 将new_person的删去
        tmp_dic = {}
        for element in result:
            this_score, this_same_person_result, this_label = element
            if this_score in tmp_dic:
                if self.new_person_str in this_label:
                    continue
                else:
                    tmp_dic[this_score] = element
            else:
                tmp_dic[this_score] = element
        result = tmp_dic.values()
        return result


    def evaluate_result(self, result):
        '''
            :param result: [(cos_sim, same_person_result, label),
                            (cos_sim, same_person_result, label),
                            (cos_sim, same_person_result, label)]
            :return: this_id(Must_same, Must_not_same, May_same), this_label(人名)
        '''
        for index, element in enumerate(result):
            this_score, this_same_person_result, this_label = element
            if this_same_person_result == self.verification_same_person and this_score > self.same_pic_threshold:
                return self.same_pic_id, this_label
            if this_same_person_result == self.verification_same_person and this_score > self.upper_threshold:
                return self.must_be_same_id, this_label
            if this_same_person_result == self.verification_same_person and this_score > self.lower_threshold:
                return self.maybe_same_id, this_label
        return self.must_be_not_same_id, ''


    def check_face_img(self, face_img, image_id):
        # 计算角度
        '''
        :param face_img: 人脸对应的矩阵
        :param image_id: 图片id
        :return: 是否进行识别(False:不进行识别)
        '''
        # 姿势检测

        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')

        face_img_str = base64.b64encode(msgpack_numpy.dumps(face_img))
        request = {
            "request_type": 'check_pose',
            "face_img_str": face_img_str,
            "image_id": image_id,
        }
        url = "http://%s:%d/" % (check_ip, check_port)
        result = image_request(request, url)
        try:
            pose_predict = json.loads(result)["pose_predict"]
            if not pose_predict:  # 加载失败
                log_file.write('\t'.join(map(str, [image_id, 'pose filter request'])) + '\n')
                log_file.close()
                return False
            else:
                pose_predict = msgpack_numpy.loads(base64.b64decode(pose_predict))
                if pose_predict == None:
                    log_file.write('\t'.join(map(str, [image_id, 'pose filter detect'])) + '\n')
                    log_file.close()
                    return False
                pitch, yaw, roll = pose_predict[0]
                if math.fabs(pitch) < self.pitch_threshold and \
                        math.fabs(yaw) < self.yaw_threshold and \
                        math.fabs(roll) < self.roll_threshold:
                    log_file.close()
                    return True
                else:
                    log_file.write('\t'.join(map(str, [image_id, 'pose filter threshold'])) + '\n')
                    log_file.close()
                    return False
        except:
            traceback.print_exc()
            log_file.close()
            return False


    def recognize_online_cluster(self, image, image_id):
        '''
            :param image: 将得到的图片进行识别,加入的LSH Forest,根据距离计算proba(不同的距离对应不同的准确率,根据已有的dist计算阈值);
                            和已经设定的阈值判断是不是一个新出现的人,确定是原来已有的人,还是不确定是原来已有的人
            # 增加统计的功能, 方便以后计算过滤原因和比例, 以及识别比例(same, not_same, maybe_same)
            :return:
        '''
        start = time.time()
        need_add = False
        has_save_num = 0
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')
        log_file.write('\t'.join(map(str, ["receive image", image_id, time.time()])) + '\n')
        try:
            image = base64.decodestring(image)
            image = zlib.decompress(image)
            im = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1)
            time_slot = get_time_slot(image_id)
            if time_slot == None:
                time_slot = 'error'
            time_slot_dir = os.path.join(tmp_face_dir, time_slot)
            if not os.path.exists(time_slot_dir):
                os.makedirs(time_slot_dir)
            tmp_pic_path = os.path.join(time_slot_dir, image_id+'.jpg')
            cv2.imwrite(tmp_pic_path, im)
            blur_result = is_blur(im)
            blur_sign, blur_var = blur_result
            if blur_sign:
                log_file.write('\t'.join(map(str, ['stat', 'blur_filter', blur_var, image_id]))+'\n')
                log_file.close()
                return self.unknown, 1.0, self.has_save_pic_feature, need_add
            align_face_img = align_face(tmp_pic_path)
            if align_face_img == None:
                log_file.write('\t'.join(map(str, ['stat', 'detect_filter', blur_var, image_id])) + '\n')
                log_file.close()
                return self.unknown, 1.0, self.has_save_pic_feature, need_add
            else:
                # 使用重新检测并对对齐的人脸进行识别
                im = align_face_img
            # 对检测到的人脸重新进行模糊检测
            blur_result = is_blur(im)
            blur_sign, blur_var = blur_result
            if blur_sign:
                log_file.write('\t'.join(map(str, ['stat', 'blur_filter', blur_var, image_id]))+'\n')
                log_file.close()
                return self.unknown, 1.0, self.has_save_pic_feature, need_add
            need_process = self.check_face_img(im, image_id)
            if not need_process:
                log_file.write('\t'.join(map(str, ['stat', 'pose_filter', blur_var, image_id])) + '\n')
                log_file.close()
                return self.unknown, 1.0, self.has_save_pic_feature, need_add
            im = cv2.resize(im, (PIC_SHAPE[1], PIC_SHAPE[2]), interpolation=cv2.INTER_LINEAR)
            im = im[:, :, ::-1]*1.0
            im = im - avg
            im = im.transpose((2, 0, 1))
            im = im[None, :]
        except:
            traceback.print_exc()
            return self.unknown, 1.0, self.has_save_pic_feature, need_add
        try:
            # 流程 : 找距离最近的图片 ; 计算prob ; 在线聚类 ; 加入LSH Forest
            im_feature = extract_feature_from_numpy(im)
            try:
                # nearest_sim_list的格式和dist_label_list的格式一样,这样可以将两个list合并,一起计算(这样不用考虑时间的因素)
                # 在识别出人名后将人名和feature放入到self.nearest
                nearest_sim_list = self.cal_nearest_sim(current_feature=im_feature)
            except:
                traceback.print_exc()
                nearest_sim_list = []
            log_file.write('\t'.join(map(str, ['nearest_sim_list :', map(str, nearest_sim_list)])) + '\n')

            # 找距离最近的图片 --- 用LSH Forest 找出最近的10张图片,然后分别计算距离
            dist_label_list = self.find_k_neighbors_with_lsh(im_feature)
            dist_label_list.extend(nearest_sim_list)
            dist_label_list = self.filter_result(dist_label_list)
            dist_label_list.sort(key=lambda x: x[0], reverse=True)
            # 计算
            if dist_label_list == None:
                this_id = self.must_be_not_same_id
                this_label = self.new_person_str + str(self.current_new_person_id)
            else:
                # 计算prob --- 根据距离计算prob
                this_id, this_label = self.evaluate_result(dist_label_list)
            # 在线聚类 --- 根据dist确定是重新增加一个人还是加入到已有的人中
            log_file.write('\t'.join(map(str, ['stat', 'recognize_id', blur_var, this_id])) + '\n')
            if dist_label_list != None and len(dist_label_list) > 0:
                log_file.write('\t'.join(map(str, ['dist_label_list :', map(str, dist_label_list)])) + '\n')
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
                log_file.write('\t'.join(map(str, ['error para :', this_id])) + '\n')
            if need_save:
                try:
                    if not os.path.exists(this_person_folder):
                        os.makedirs(this_person_folder)
                        os.chmod(this_person_folder, stat.S_IRWXG + stat.S_IRWXO + stat.S_IRWXU)
                    this_pic_name = os.path.join(this_person_folder, image_id+'.png')
                    imsave(this_pic_name, np.transpose(im[0], (1, 2, 0)))
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
                log_file.write('\t'.join(map(str, ['stat recognize_time :', (end - start), 'this_id :', self.trans_dic.get(this_id)])) + '\n')
                log_file.close()
                return this_label.replace(self.must_same_str, ''), \
                       str(dist_label_list[0][0]), str(has_save_num), str(need_add)
            else:
                # 灰度区域,不显示人名
                end = time.time()
                log_file.write('\t'.join(map(str, ['gray area recog time :',(end - start)])) + '\n')
                log_file.close()
                # return this_label.replace(self.maybe_same_str, ''), \
                #        str(dist_label_list[0][0]), str(has_save_num), str(need_add)
                return self.unknown, str(dist_label_list[0][0]), str(has_save_num), str(need_add)
        except:
            traceback.print_exc()
            log_file.close()
            return self.unknown, str(100.0), str(has_save_num), str(False)


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        request_type = self.get_body_argument('request_type')
        if request_type == 'recognization':
            try:
                image_id = self.get_body_argument("image_id")
                image = self.get_body_argument("image")
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

