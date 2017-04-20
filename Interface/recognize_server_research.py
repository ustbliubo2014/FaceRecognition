# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: recognize_server_search.py
@time: 2016/8/30 15:56
@contact: ustb_liubo@qq.com
@annotation: recognize_server_research
"""
import sys
import numpy as np
import time
import json
import base64
import tornado.ioloop
import tornado.web
import traceback
import os
import msgpack
from collections import Counter
import msgpack_numpy
from recog_util import image_request, get_time_slot, get_current_day
import pdb
from sklearn.neighbors import LSHForest
from MyThread import MyThread
import zlib
import logging
from logging.config import fileConfig
import sklearn.metrics.pairwise as pw
import cv2
from collections import deque
from recog_util import is_blur
from research_model import (lower_verif_threshold, verification_same_person, verification_model, port, find_big_face,
                        extract_feature_from_binary_data, upper_verif_threshold, FEATURE_DIM, nearest_num, same_pic_threshold)

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')

size_threshold = 120
model_label = 'research'
tmp_face_dir = '/tmp/research_face'
log_dir = model_label+'_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class FaceRecognition():
    def __init__(self):
        self.unknown = ''
        self.same_person_num = 1
        self.has_cal_dist = []
        self.NeighbourNum = 10
        # 如果管理员加载图片, 把图片放到all_pic_data_folder下指定人的目录(图片文件和特征文件的文件名相同)
        self.all_pic_feature_data_folder = '/data/liubo/face/research_feature_self'     # 研究院的模型直接存储特征
        # 保存图片可以方便以后查看效果, 方便前端显示, 也方便管理员进行标注
        self.all_pic_data_folder = '/data/liubo/face/research_self'
        if not os.path.exists(self.all_pic_data_folder):
            os.makedirs(self.all_pic_data_folder)
        if not os.path.exists(self.all_pic_feature_data_folder):
            os.makedirs(self.all_pic_feature_data_folder)
        self.n_neighbors = 10
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
        self.upper_threshold = upper_verif_threshold
        self.lower_threshold = lower_verif_threshold
        self.same_pic_threshold = same_pic_threshold
        self.trans_dic = {self.same_pic_id: 'same_pic', self.must_be_same_id: 'must_same_id',
                          self.must_be_not_same_id: 'must_not_same_id', self.maybe_same_id: 'maybe_same_id'}
        self.nearest = deque(maxlen=nearest_num)
        self.verification_same_person = 0


    def cal_nearest_sim(self, current_feature):
        nearest_sim_list = []
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')

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
                    log_file.write('cal_nearest_sim error'+'\n')
                    traceback.print_exc()
                    continue
            return nearest_sim_list
        except:
            traceback.print_exc()
            return nearest_sim_list


    def load_train_data(self, data_folder):
        # 直接读取图片特征, 返回所有特征和label
        all_pic_feature = []
        all_label = []
        person_list = os.listdir(data_folder)
        for person in person_list:
            if person == self.unknown or self.must_same_str in person or self.maybe_same_str in person:
                continue
            person_path = os.path.join(data_folder, person)
            pic_feature_list = os.listdir(person_path)
            for pic_feature_path in pic_feature_list:
                pic_feature = msgpack_numpy.load(open(os.path.join(person_path, pic_feature_path), 'rb'))
                all_pic_feature.append(pic_feature)
                all_label.append(person)
        all_pic_feature = np.asarray(all_pic_feature)
        all_label = np.asarray(all_label)
        return all_pic_feature, all_label


    def find_current_new_person_id(self):
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')

        old_person_id = []
        # 保存的是原始图片
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


    def extract_pic_feature(self, pic_path):
        try:
            result = extract_feature_from_binary_data(open(pic_path, 'rb'))
            if result == None:
                return
            face_num, all_frames, all_feature = result
            biggest_face_index = find_big_face(all_frames)
            pic_frame = all_frames[biggest_face_index]
            pic_feature = all_feature[biggest_face_index]
            x, y, width, height = pic_frame
            face_pic = cv2.imread(pic_path)[y:y+width, x:x+height, :]
            return face_pic, pic_feature
        except:
            traceback.print_exc()
            return None


    def load_all_data(self):
        # 将以前标记的数据全部读入(直接读入的是特征), 用LSH Forest保存,方便计算距离
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')
        try:
            all_pic_feature, all_label = self.load_train_data(self.all_pic_feature_data_folder)
            train_label = np.asarray(all_label)
            if len(all_pic_feature) == len(train_label) and len(train_label) > 0:
                start = time.time()
                self.lshf.fit(all_pic_feature, train_label)
                self.all_pic_feature = list(all_pic_feature)
                self.all_labels = list(train_label)
                end = time.time()
                self.load_time = end
                self.user_count = Counter(self.all_labels)
                log_file.write('\t'.join(map(str, [self.user_count,
                                           'fit all data time :', (end - start)]))+'\n')
                log_file.close()
        except:
            traceback.print_exc()
            log_file.close()
            return


    def save_pic_feature(self, pic_path, person_name):
        #  将已经存在的文件生成特征并保存到指定文件夹下, 用于管理员加入新的图片(加入新的图片后, 提取特征, 保存到指定文件夹)
        person_pic_folder_path = os.path.join(self.all_pic_data_folder, person_name)
        person_feature_folder_path = os.path.join(self.all_pic_feature_data_folder, person_name)
        if not os.path.exists(person_pic_folder_path):
            os.makedirs(person_pic_folder_path)
        if not os.path.exists(person_feature_folder_path):
            os.makedirs(person_feature_folder_path)
        pic_name = os.path.split(pic_path)[-1]
        # 特征文件
        person_feature_path = os.path.join(person_feature_folder_path, pic_name)
        # 人脸文件
        person_pic_path = os.path.join(person_pic_folder_path, pic_name)
        result = extract_feature_from_binary_data(open(pic_path, 'rb'))
        if result == None:
            return
        face_num, all_frames, all_feature = result
        biggest_face_index = find_big_face(all_frames)
        pic_frame = all_frames[biggest_face_index]
        pic_feature = all_feature[biggest_face_index]
        x, y, width, height = pic_frame
        face_pic = cv2.imread(pic_path)[y:y+width, x:x+height, :]
        cv2.imwrite(person_pic_path, face_pic)
        msgpack_numpy.dump(pic_feature, open(person_feature_path, 'wb'))


    def add_all_new_pic(self):
        '''
            将从上次加载数据到当前新增的文件都加载到LSH Forest(有可能是新增加一个人,还有可能是对已有的人增加新图片)
            遍历文件夹(self.all_pic_feature_data_folder), 根据文件的时间判断是否需要加入该图片的特征
            系统在管理员标注图片后, 将人脸图片和特征文件同时进行移动, 所以现在只需要将特征和对应的label加入LSH就可以了
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
                    request = {
                        "label": person,
                        "request_type": 'add',
                        "one_pic_feature": pic_path
                    }
                    url = "http://127.0.0.1:%d/"%port
                    result = image_request(request, url)
                    try:
                        add_flag = json.loads(result)["add"]
                        if not add_flag:    # 加载失败
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
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')
        try:
            # 读入数据时已经转换成需要的尺寸
            result = self.extract_pic_feature(pic_path)
            if result == None:
                return False
            face_pic, pic_feature = result
            self.add_one_pic(pic_feature, label)
            pic_name = os.path.split(pic_path)[1]
            this_person_pic_folder = os.path.join(self.all_pic_data_folder, label)
            this_person_feature_folder = os.path.join(self.all_pic_feature_data_folder, label)
            if not os.path.exists(this_person_pic_folder):
                os.makedirs(this_person_pic_folder)
            if not os.path.exists(this_person_feature_folder):
                os.makedirs(this_person_feature_folder)
            # 直接存储图片对应的特征, 同时保存图片文件
            this_pic_feature_name = os.path.join(this_person_feature_folder, pic_name + '.p')
            msgpack_numpy.dump(pic_feature, open(this_pic_feature_name, 'wb'))
            this_pic_face_name = os.path.join(this_person_pic_folder, pic_name + '.jpg')
            cv2.imwrite(this_pic_face_name, face_pic)
            log_file.write('\t'.join(map(str, [pic_path, this_pic_face_name]))+'\n')
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
        one_pic_feature = np.asarray(one_pic_feature)
        self.lshf.partial_fit(one_pic_feature.reshape(1, FEATURE_DIM), pic_label)
        self.all_labels.append(pic_label)
        self.all_pic_feature.append(np.reshape(one_pic_feature, newshape=(1, one_pic_feature.size)))


    def find_k_neighbors_with_lsh(self, one_pic_feature):
        '''
            :param one_pic_feature: 图像特征
            :return: 需要返回neighbors的特征,用于计算pariwise
        '''
        try:
            one_pic_feature = np.asarray(one_pic_feature)
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
            try:
                this_score, this_same_person_result, this_label = element
                this_score = float(this_score)
                if this_score in tmp_dic:
                    if self.new_person_str in this_label:
                        continue
                    else:
                        tmp_dic[this_score] = element
                else:
                    tmp_dic[this_score] = element
            except:
                traceback.print_exc()
                continue
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


    def recognize_online_cluster(self, image, image_id):
        '''
            :param image: 将得到的图片进行识别,加入的LSH Forest,根据距离计算proba(不同的距离对应不同的准确率,根据已有的dist计算阈值);
                            和已经设定的阈值判断是不是一个新出现的人,确定是原来已有的人,还是不确定是原来已有的人
            :return:
        '''
        start = time.time()
        need_add = False
        need_save = False
        current_day = get_current_day()
        log_file = open(os.path.join(log_dir, current_day+'.txt'), 'a')
        log_file.write('\t'.join(map(str, ["receive image", image_id, time.time()])) + '\n')
        feature_str = ''
        try:
            image = base64.decodestring(image)
            image = zlib.decompress(image)
            im = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1)
            log_file.write('\t'.join(map(str, ['shape :', im.shape[0], im.shape[1]])) + '\n')
            # 图片尺寸过滤
            if im.shape[0] < size_threshold or im.shape[1] < size_threshold:
                log_file.write('\t'.join(map(str, ['stat recognize_time :', (time.time() - start), 'small_size'])) + '\n')
                log_file.close()
                return self.unknown, 1.0, feature_str, need_save
            # 清晰度过滤
            blur_sign, blur_var = is_blur(cv2.resize(im, (96, 96)))
            if blur_sign:
                log_file.write('\t'.join(map(str, ['stat recognize_time :', (time.time() - start), 'blur_filter', blur_var])) + '\n')
                log_file.close()
                return self.unknown, 1.0, feature_str, need_save
            #  保存传过来的图片
            # img_file = '/tmp/research_face/%s.jpg' %image_id
            time_slot = get_time_slot(image_id)
            if time_slot == None:
                time_slot = 'error'
            time_slot_dir = os.path.join(tmp_face_dir, time_slot)
            if not os.path.exists(time_slot_dir):
                os.makedirs(time_slot_dir)
            img_file = os.path.join(time_slot_dir, image_id+'.jpg')
            cv2.imwrite(img_file, im)
        except:
            traceback.print_exc()
            log_file.close()
            return self.unknown, 1.0, feature_str, need_save
        try:
            # 流程 : 找距离最近的图片 ; 计算prob ; 在线聚类 ; 加入LSH Forest
            result = self.extract_pic_feature(img_file)
            if result == None:
                log_file.write('\t'.join(map(str, ['stat not_find_face', 'time :', (time.time() - start)]))+'\n')
                log_file.close()
                return self.unknown, 1.0, feature_str, need_save
            face_pic, im_feature = result

            try:
                # nearest_sim_list的格式和dist_label_list的格式一样,这样可以将两个list合并,一起计算(这样不用考虑时间的因素)
                # 在识别出人名后将人名和feature放入到self.nearest
                nearest_sim_list = self.cal_nearest_sim(current_feature=im_feature)
            except:
                traceback.print_exc()
                nearest_sim_list = []
            log_file.write('\t'.join(map(str, ['nearest_sim_list :', map(str, nearest_sim_list)])) + '\n')
            feature_str = base64.b64encode(msgpack_numpy.dumps(im_feature))
            log_file.write('\t'.join(map(str, ['extract_feature_time :', (time.time() - start)]))+'\n')
            # 找距离最近的图片 --- 用LSH Forest 找出最近的10张图片,然后分别计算距离

            tmp_list = self.find_k_neighbors_with_lsh(im_feature)
            nearest_sim_list.sort(key=lambda x: x[0], reverse=True)
            nearest_sim_list.extend(tmp_list)
            dist_label_list = nearest_sim_list[:]

            # 计算
            log_file.write('\t'.join(map(str, ['dist_label_list :', map(str, dist_label_list)])) + '\n')
            if dist_label_list == None:
                this_id = self.must_be_not_same_id
                this_label = self.new_person_str + str(self.current_new_person_id)
            else:
                # 计算prob --- 根据距离计算prob
                this_id, this_label = self.evaluate_result(dist_label_list)
            # 不管概率, 都要将最新的一张图片加入到self.nearest
            self.nearest.append((this_label, im_feature))
            log_file.write('\t'.join(map(str, ['self.nearest :', map(str, self.nearest)])) + '\n')
            # 在线聚类 --- 根据dist确定是重新增加一个人还是加入到已有的人中
            if this_id == self.same_pic_id:
                need_add = False
            elif this_id == self.must_be_same_id:
                need_add = False
                need_save = True
                this_person_pic_folder = os.path.join(self.all_pic_data_folder, this_label+self.must_same_str)
                this_person_feature_folder = os.path.join(self.all_pic_feature_data_folder, this_label+self.must_same_str)
            elif this_id == self.must_be_not_same_id:
                this_label = self.new_person_str + str(self.current_new_person_id)
                self.current_new_person_id += 1
                this_person_pic_folder = os.path.join(self.all_pic_data_folder, this_label)
                this_person_feature_folder = os.path.join(self.all_pic_feature_data_folder, this_label)
                need_add = True
                need_save = True
            elif this_id == self.maybe_same_id:
                this_person_pic_folder = os.path.join(self.all_pic_data_folder, this_label + self.maybe_same_str)
                this_person_feature_folder = os.path.join(self.all_pic_feature_data_folder, this_label + self.maybe_same_str)
                need_add = False # prob在灰度区域的不如入,其余情况加入
                need_save = True
            else:
                log_file.write('\t'.join(map(str, ['error para :', this_id]))+'\n')
            if need_save:
                try:
                    if not os.path.exists(this_person_pic_folder):
                        os.makedirs(this_person_pic_folder)
                    if not os.path.exists(this_person_feature_folder):
                        os.makedirs(this_person_feature_folder)
                    # 直接存储图片对应的特征, 同时保存图片文件
                    this_pic_feature_name = os.path.join(this_person_feature_folder, image_id+'.p')
                    msgpack_numpy.dump(im_feature, open(this_pic_feature_name, 'wb'))
                    this_pic_face_name = os.path.join(this_person_pic_folder, image_id+'.jpg')
                    cv2.imwrite(this_pic_face_name, face_pic)
                except:
                    traceback.print_exc()
                    return self.unknown, 1.0, feature_str, False
            # 加入LSH Forest --- partial_fit
            if need_add:
                self.add_one_pic(im_feature, this_label)
                # 根据label和image_id可以存生成文件名,确定是否要存储文件[可以选择在服务器和本地同时存储]
            if this_id == self.same_pic_id or this_id == self.must_be_not_same_id or this_id == self.must_be_same_id:
                end = time.time()
                log_file.write('\t'.join(map(str, ['stat recognize_time :',(end - start), 'this_id :', self.trans_dic.get(this_id)]))+'\n')
                log_file.close()
                need_save = True
                return this_label.replace(self.must_same_str, ''), str(dist_label_list[0][0]), str(feature_str), str(need_save)
            else:
                # 灰度区域,不显示人名
                end = time.time()
                log_file.write('\t'.join(map(str, ['stat gray_area :',(end - start)]))+'\n')
                log_file.close()
                return self.unknown, str(dist_label_list[0][0]), str(feature_str), str(False)
        except:
            traceback.print_exc()
            log_file.close()
            return self.unknown, str(100.0), str(feature_str), str(False)


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
        elif request_type == 'add':     # 向LSH Forest中加入新图片
            one_pic_path = self.get_body_argument("one_pic_feature")
            label = self.get_body_argument("label")
            # 传入的是base64的图片
            self.write(json.dumps({"add": face_recognition.add_one_new_pic(one_pic_path, label)}))


def server(application):
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()


def add_new_pic(face_recognization):
    while True:
        # 直接请求本地的服务,加入LSH Forest
        time.sleep(10)
        face_recognization.add_all_new_pic()


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

