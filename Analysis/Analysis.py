# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: Analysis.py
@time: 2016/11/16 10:17
@contact: ustb_liubo@qq.com
@annotation: Analysis
"""
import sys
import os
from sklearn.neighbors import LSHForest
import time
from collections import deque, Counter
import numpy as np
import traceback
import sklearn.metrics.pairwise as pw
import base64
import cv2
import requests
from util import get_current_day, get_current_time
import cPickle
import pdb
from sql_operator import get_all_new_face, get_all_name, get_all_annotate_half
import msgpack_numpy
import base64
from optparse import OptionParser
from conf import SkyEyeConf, ResearchConf
import json
from sql_operator import insert_pic_list

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


class Analysis():
    def __init__(self, conf):
        self.unknown = ''
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
        self.user_count = {}
        self.nearest_num = 5
        # 只保存最近15秒的图片
        self.nearest_time_threshold = 15
        self.feature_url = conf.feature_url
        # 以后调模型时不在修改最后一个卷积层的维度
        self.feature_dim = conf.feature_dim
        # 每次更换模型的时候需要修改这两个参数
        self.same_pic_threshold = conf.same_pic_threshold
        self.upper_threshold = conf.upper_threshold
        self.lower_threshold = conf.lower_threshold
        self.pitch_threshold = 20
        self.yaw_threshold = 20
        self.roll_threshold = 20
        self.max_dist_threshold = 100
        #  [(label, feature),...,(label, feature)]
        self.nearest = deque(maxlen=self.nearest_num)
        self.trans_dic = {self.must_be_same_id: 'must_same_id', self.same_pic_id: 'same_pic',
                self.must_be_not_same_id: 'not_same_id', self.maybe_same_id: 'maybe_same_id'}
        self.all_feature_label_file = conf.all_feature_label_file
        self.log_dir = conf.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tmp_jpg_file = 'tmp.jpg'
        self.model_label = conf.model_label


    def find_current_new_person_id(self):
        all_id_name = get_all_name()
        current_new_person_id = -1
        for id_name in all_id_name:
            name = id_name[1]
            if name.startswith(self.new_person_str):
                current_new_person_id = max(current_new_person_id, int(name.replace(self.new_person_str, '')))
        current_new_person_id = current_new_person_id + 1
        return current_new_person_id


    def cal_nearest_sim(self, current_time, current_feature):
        nearest_sim_list = []
        try:
            length = len(self.nearest)
            for k in range(length):
                this_label, pre_feature, pre_time = self.nearest[k]
                if current_time - pre_time > self.nearest_time_threshold:
                    continue
                this_sim = pw.cosine_similarity(pre_feature, current_feature)
                nearest_sim_list.append((this_sim, this_label))
            return nearest_sim_list
        except:
            traceback.print_exc()
        return nearest_sim_list


    def extract_pic_feature(self, face_array):
        '''
            # 传入半身照片,得到人脸照片(必须要做检测,因为有可能会更新检测模型,导致识别不准)
            # 用于人工添加图片加到LSHForest
            # 仍然使用人脸识别的接口, 解析得到的特征
            :param face_array: 人脸图片(numpy格式)
            :return:face_frame, feature(numpy格式)
        '''
        try:
            cv2.imwrite(self.tmp_jpg_file, face_array)
            result = requests.post(self.feature_url, open(self.tmp_jpg_file, 'rb').read())
            if result.status_code == 200:
                try:
                    content = result.content
                    tmp = content.split('\n')
                    if len(tmp) < 3:
                        return None, None
                    face_num = int(tmp[0].split(':')[1])
                    if face_num == 1:
                        frame = map(float, tmp[1].split(','))
                        feature = map(float, tmp[2].split(',')[:-1])
                        if np.sum(feature) == 0:
                            print 'filter'
                            return None, None
                        return frame, feature
                except:
                    traceback.print_exc()
                    return None, None
            else:
                return None, None
        except:
            traceback.print_exc()
            return None, None



    def load_all_data(self):
        # 将以前标记的数据全部读入,用LSH Forest保存,方便计算距离
        # 使用半身照进行检测和识别(输入图片,得到content,解析content得到feature)
        current_day = get_current_day()
        log_file = open(os.path.join(self.log_dir, current_day + '.txt'), 'a')
        if not os.path.exists(self.all_feature_label_file):
            return
        start = time.time()
        # 从数据库中得到半身照和人名
        half_pic_name_list = get_all_annotate_half()
        for element in half_pic_name_list:
            image, name = element
            im = cv2.imdecode(np.fromstring(base64.decodestring(image), dtype=np.uint8), 1)
            tmp_1 = self.extract_pic_feature(im)
            if tmp_1 == None:
                continue
            face_frame, im_feature = tmp_1
            if im_feature == None or face_frame == None:
                continue
            if np.sum(im_feature) == 0:
                print im.shape, name, 'blur'
                continue
            print im.shape, name
            im_feature = list(im_feature)
            # type(im_feature) < type 'list' > ;  len(im_feature) 256
            this_label = name
            self.all_pic_feature.append(im_feature)
            self.all_labels.append(this_label)
            self.lshf.partial_fit(im_feature, this_label)
        end = time.time()
        self.user_count = Counter(self.all_labels)
        current_time = get_current_time()
        log_file.write('\t'.join(map(str, [current_time, self.user_count, 'fit all data time :', (end - start)])) + '\n')
        log_file.close()


    def add_one_pic(self, one_pic_feature, pic_label):
        '''
            将一个图像的特征加入到LSH Forest,同时将对应的标签加入到self.all_labels
            :param pic_feature: array shape :(1,512)
            :param pic_label: (1,)
            :return:
        '''
        self.lshf.partial_fit(one_pic_feature.reshape(1, self.feature_dim), pic_label)
        self.all_labels.append(pic_label)
        self.all_pic_feature.append(np.reshape(one_pic_feature, newshape=(1, one_pic_feature.size)))


    def add_all_new_pic(self):
        '''
            遍历数据库(将修改过的数据加入LSHForest)
            一分钟一次(避免频繁查数据库, 也不会造成太大的延迟)
            使用研究院的模型时, 只能先保存特征, 直接移动特征(在数据库中加一列)
        '''
        current_day = get_current_day()
        log_file = open(os.path.join(self.log_dir, current_day + '.txt'), 'a')
        start = time.time()
        add_num = 0
        all_new_pic_name = get_all_new_face()
        for feature_str, person_name in all_new_pic_name:
            face_feature = np.reshape(msgpack_numpy.loads(base64.b64decode(feature_str)), (1, self.feature_dim))
            self.add_one_pic(face_feature, person_name)
            add_num += 1
        if add_num > 0:
            end = time.time()
            current_time = get_current_time()
            log_file.write('\t'.join(map(str, [current_time, 'add_pic_num :', add_num, 'Dynamic_increase_time :', (end - start)])) + '\n')
            log_file.close()
        else:
            log_file.close()


    def find_k_neighbors_with_lsh(self, one_pic_feature):
        '''
            :param one_pic_feature: 图像特征
            :return: 需要返回neighbors的特征, 用于计算pariwise
        '''
        try:
            one_pic_feature = one_pic_feature.reshape(1, self.feature_dim)
            tmp = self.lshf.kneighbors(one_pic_feature, n_neighbors=self.n_neighbors, return_distance=True)
            neighbors_label = np.asarray(self.all_labels)[tmp[1][0]]
            neighbors_feature = np.asarray(self.all_pic_feature)[tmp[1][0]]
            cos_sim_list = []
            for index in range(len(neighbors_feature)):
                pair_score = pw.cosine_similarity(neighbors_feature[index], one_pic_feature)[0][0]
                cos_sim_list.append(pair_score)
            result = zip(cos_sim_list, neighbors_label)
            result = self.filter_result(result)
            result.sort(key=lambda x: x[0], reverse=True)
            return result
        except:
            traceback.print_exc()
            return None


    def filter_result(self, result):
        '''
            :param result: [(cos_sim, label), (cos_sim, label), (cos_sim, label)] 按cos_sim降序排列
            :return: this_id(Must_same, Must_not_same, May_same), this_label(人名)
        '''
        # 分值相同的, 将new_person的删去
        tmp_dic = {}
        for element in result:
            this_score, this_label = element
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
                    程序中只根据cos_sim做判断, 不在使用same_person_result
            :return: this_id(Must_same, Must_not_same, May_same), this_label(人名)
        '''
        for index, element in enumerate(result):
            this_score, this_label = element
            if this_score > self.same_pic_threshold:
                return self.same_pic_id, this_label
            if this_score > self.upper_threshold:
                return self.must_be_same_id, this_label
            if this_score > self.lower_threshold:
                return self.maybe_same_id, this_label
        return self.must_be_not_same_id, ''


    def recognize_one_feature(self, im_feature, image_id):
        '''
            根据特征确定label
            :param image_id : 大图的文件名+face_id(第几个人脸) --- 方便定位
        '''
        start = time.time()
        feature_str = base64.b64encode(msgpack_numpy.dumps(im_feature))
        # im_feature = msgpack_numpy.loads(base64.b64decode(feature_str))
        current_day = get_current_day()
        log_file = open(os.path.join(self.log_dir, current_day + '.txt'), 'a')
        current_time = get_current_time()
        log_file.write('\t'.join(map(str, [current_time, "receive image", image_id])) + '\n')
        try:
            # 流程 : 找距离最近的图片 ; 计算prob ; 在线聚类 ; 加入LSH Forest
            try:
                current_time = float(image_id)
                nearest_sim_list = self.cal_nearest_sim(current_time=current_time, current_feature=im_feature)
                # print 'current_time :', current_time, 'nearest_sim_list :', nearest_sim_list
            except:
                traceback.print_exc()
                nearest_sim_list = []
            # 找距离最近的图片 --- 用LSH Forest 找出最近的10张图片,然后分别计算距离
            dist_label_list = self.find_k_neighbors_with_lsh(im_feature)
            dist_label_list.extend(nearest_sim_list)
            dist_label_list = self.filter_result(dist_label_list)
            dist_label_list.sort(key=lambda x: x[0], reverse=True)

            # 计算
            if dist_label_list == None:
                # 不考虑new_person的情况,小于阈值的都判断为new_person
                this_id = self.must_be_not_same_id
                this_label = 'new_person'
                # this_id = self.must_be_not_same_id
                # this_label = self.new_person_str + str(self.current_new_person_id)
            else:
                # 计算prob --- 根据距离计算prob
                this_id, this_label = self.evaluate_result(dist_label_list)
            # 在线聚类 --- 根据dist确定是重新增加一个人还是加入到已有的人中
            if dist_label_list != None and len(dist_label_list) > 0:
                current_time = get_current_time()
                log_file.write('\t'.join(map(str, [current_time, 'dist_label_list :', map(str, dist_label_list)])) + '\n')
            # need_add 决定是否加入LSHForest ;  need_save决定是否存入数据库
            if this_id == self.same_pic_id:
                need_add = False
                need_save = True
            elif this_id == self.must_be_same_id:
                need_add = False
                need_save = True
            elif this_id == self.must_be_not_same_id:
                # 现在的版本不用加入新人, 不能识别的全部返回new_person
                this_label = 'new_person'
                need_save = True
                need_add = False
                # this_label = self.new_person_str + str(self.current_new_person_id)
                # self.current_new_person_id += 1
                # need_add = True
                # need_save = True
            elif this_id == self.maybe_same_id:
                need_add = False
                need_save = False
            else:
                current_time = get_current_time()
                log_file.write('\t'.join(map(str, [current_time, 'error para :', this_id])) + '\n')
                return self.unknown, str(self.max_dist_threshold), feature_str, str(False)
            self.nearest.append((this_label, im_feature, image_id))
            # 现在不在增加new_person
            # # 加入LSH Forest --- partial_fit
            # if need_add:
            #     # 只将新人的图片加入LSHForest并保存到文件
            #     self.add_one_pic(im_feature, this_label)
            #     write_start = time.time()
            #     tmp_file = open(self.all_feature_label_file, 'a')
            #     tmp_file.write(base64.b64encode(msgpack_numpy.dumps((im_feature, this_label)))+'\n')
            #     tmp_file.close()
            #     print 'write time :', (time.time() - write_start)
            #     # 根据label和image_id可以存生成文件名,确定是否要存储文件[可以选择在服务器和本地同时存储]
            # 统计有多少图片在gray area
            log_file.write('\t'.join(map(str, ['stat', 'recognize_id', self.trans_dic[this_id], 'recog time :', (time.time() - start)])) + '\n')
            log_file.close()
            if this_id == self.same_pic_id or this_id == self.must_be_not_same_id or this_id == self.must_be_same_id:
                if this_label == None or dist_label_list == None:
                        # 数据库里可能一个人也没有, 这时this_label = None
                        return self.unknown, str(self.max_dist_threshold), feature_str, str(False)
                else:
                    return this_label.replace(self.must_same_str, ''), str(dist_label_list[0][0]), feature_str, str(need_save)
            else:
                # 灰度区域,不显示人名
                # return this_label.replace(self.maybe_same_str, ''), tr(dist_label_list[0][0]), str(has_save_num), str(need_add)
                return self.unknown, str(dist_label_list[0][0]), feature_str, str(need_save)
        except:
            traceback.print_exc()
            log_file.close()
            return self.unknown, str(self.max_dist_threshold), feature_str, str(False)


    def recognize_online_cluster(self, content, image_id):
        '''
            该程序不需要存储图片, 只需要将标志返回就可以
            增加过滤,
            :param content: 检测识别返回的结果
            :return:
        '''
        tmp = content.split('\n')
        print 'len(tmp) :', len(tmp)
        if len(tmp) < 3:
            return None
        face_num = int(tmp[0].split(':')[1])
        all_frames = []
        all_recognize_result = []
        for k in range(face_num):
            frame = map(float, tmp[2 * k + 1].split(','))
            feature = np.reshape(np.asarray(map(float, tmp[2 * k + 2].split(',')[:-1])), (1, self.feature_dim))
            person_name, score, has_save_pic_feature, need_save = self.recognize_one_feature(feature, image_id)
            all_recognize_result.append((person_name, score, has_save_pic_feature, need_save))
            all_frames.append(frame)
        return zip(all_frames, all_recognize_result)


    def offline_add(self, folder):
        # 线下自己将文件夹中的数据导入(每个图片以label命名)
        pic_list = os.listdir(folder)
        pic_info = []
        for pic in pic_list[:]:
            print 'pic :', pic
            label = pic.split('.')[0]
            label = label.decode('gbk').encode('utf-8')
            pic_path = os.path.join(folder, pic)
            img_array = cv2.imread(pic_path)
            try:
                tmp = self.extract_pic_feature(img_array)
                if tmp == None:
                    continue
                face_frame, im_feature = tmp
                if face_frame == None or im_feature == None:
                    continue
            except:
                traceback.print_exc()
                continue
            x, y, w, h = face_frame
            face = img_array[int(y):int(y + h), int(x):int(x + w), :]
            algorithm = self.model_label
            face_str = base64.encodestring(cv2.imencode('.jpg', face)[1].tostring())
            img_str = base64.encodestring(cv2.imencode('.jpg', img_array)[1].tostring())
            # tmp_array  = cv2.imdecode(np.fromstring(base64.decodestring(img_str), dtype=np.uint8), 1)
            # cv2.imwrite(str(time.time())+'.jpg', tmp_array)

            pic_info.append((label, algorithm, face_str, img_str))
        insert_pic_list(pic_info)



def main(conf):
    # # 将某个文件夹下的图片导入到数据据
    # analyse = Analysis(conf)
    # analyse.offline_add('/data/face/image-identify/Analysis/test_one')

    analyse = Analysis(conf)
    analyse.load_all_data()

    print 'finish load data', len(analyse.all_labels)

    folder = 'test_one'
    pic_list = os.listdir(folder)
    for pic in pic_list:
        try:
            pic_path = os.path.join(folder, pic)
            pic_binary_data = open(pic_path, 'rb').read()
            request = requests.post(analyse.feature_url, pic_binary_data)
            content = request.content
            image_id = time.time()
            result = analyse.recognize_online_cluster(content, str(image_id))
            print pic, result
        except:
            traceback.print_exc()
            continue
        # print len(analyse.all_labels), len(analyse.all_pic_feature), result
        # img = cv2.imread(pic_path)
        # for index, element in enumerate(result):
        #     frame, name = element
        #     x, y, w, h = frame
        #     face = img[int(y):int(y + h), int(x):int(x + w), :]
            # cv2.imwrite('face%d.jpg' % index, face)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", "--model_label", dest="model_label", help="使用的哪个模型 skyeye or research")
    (options, args) = parser.parse_args()
    model_label = options.model_label
    if model_label == 'skyeye':
        conf = SkyEyeConf()
    elif model_label == 'research':
        conf = ResearchConf()
    else:
        print 'error para'
        sys.exit()
    main(conf)
    # python Analysis.py - -model_label research
    # python Analysis.py - -model_label skyeye

