#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: search_neighbors_pictures.py
@time: 2016/6/7 16:50
@contact: ustb_liubo@qq.com
@annotation: search_neighbors_pictures
"""
from collections import Counter
import sys
import numpy as np
from sklearn.neighbors import LSHForest
import os
from scipy.misc import imread, imresize, imsave
import pdb
from load_DeepId_model import load_deepid_model
import traceback
from PIL import Image
import logging
from split_pic import get_landmarks, get_nose, get_left_eye, get_right_eye, cal_angel
from skimage.transform import rotate
import cPickle


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='search1.log',
                    filemode='a')

# pic_shape = (50, 50, 3)
# pic_shape = (156, 124, 3)
train_person_start_index = 300
# deepid模型用到的数据
clf_model_file = '/data/liubo/face/vgg_face_dataset/model/GaussianNB_model.p'

def load_train_data(folder, shape=(128, 128, 3), need_pic_list=False, pic_num_threshold=1000, part_func=None,
                    person_num=None):
    person_list = os.listdir(folder)
    data = []
    label = []
    all_pic_list = []
    label_trans_dic = {}
    current_label = 0
    if person_num != None:
        person_num = min(len(person_list), person_num)
    else:
        person_num = len(person_list)
    for person_index, person in enumerate(person_list[:person_num]):
        print person_index, person
        person_path = os.path.join(folder, person)
        if not os.path.isdir(person_path):
            continue
        pic_list = os.listdir(person_path)
        np.random.shuffle(pic_list)
        this_pic_num_threshold = min(len(pic_list), pic_num_threshold)
        pic_list = pic_list[:this_pic_num_threshold]
        if person not in label_trans_dic:
            label_trans_dic[person] = current_label
            person_id = current_label
            current_label += 1
        else:
            person_id = label_trans_dic.get(person)
        all_pic_list.extend(pic_list)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            try:
                arr = np.array(Image.open(pic_path))
                arr = imresize(arr, shape)
                if part_func != None:
                        landmarks = get_landmarks(arr)
                        angle = cal_angel(landmarks)
                        rotate_im = rotate(arr, angle)
                        rotate_im = np.array(Image.fromarray(np.uint8(rotate_im*255)))
                        new_landmarks = get_landmarks(rotate_im)
                        arr = part_func(rotate_im, new_landmarks)
                        # pdb.set_trace()
                data.append(arr)
                label.append(person_id)
                # label.append(person)
            except:
                traceback.print_exc()
                print 'error_pic_path :', pic_path
    data = np.asarray(data, dtype=np.float32)
    data = data / 255.0
    label = np.asarray(label)
    if need_pic_list:
        return data, label, all_pic_list
    else:
        return data, label


def load_batch_train_data(folder, start_person_index, shape, pic_num=10, batch_num=100, is_train=True,
                          part_func=None):
    '''
    :param folder: 原始数据集
    :param start_person_index: 和batch_num一起使用,每次读入batch_num个人的数据
    :param batch_num: 每次读入多少人的照片
    :param shape: 读入的每张图片的尺寸
    :return: data, label [label是人名,方便测试,sklearn也支持string类型的label]
    '''
    person_list = os.listdir(folder)
    person_list.sort()
    data = []
    label = []
    label_trans_dic = {}
    current_label = 0
    end_person_index = min(len(person_list), start_person_index+batch_num)
    for person in person_list[start_person_index:end_person_index]:
        if person == 'unknown':
            continue
        logging.debug(person)
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        # if len(pic_list) < 5:
        #     continue
        pic_list.sort()
        if is_train:
            pic_list = pic_list[:pic_num]
        else:
            pic_list = pic_list[pic_num:pic_num+10]
            # 测试时每个人使用10张图片
        if person not in label_trans_dic:
            label_trans_dic[person] = current_label
            person_id = current_label
            current_label += 1
        else:
            person_id = label_trans_dic.get(person)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            try:
                arr = np.array(Image.open(pic_path))
                arr = imresize(arr, shape)
                if part_func != None:
                    landmarks = get_landmarks(arr)
                    angle = cal_angel(landmarks)
                    rotate_im = rotate(arr, angle)
                    rotate_im = np.array(Image.fromarray(np.uint8(rotate_im*255)))
                    new_landmarks = get_landmarks(rotate_im)
                    arr = part_func(rotate_im, new_landmarks)
                data.append(arr)
                # label.append(person_id)
                label.append(person)
            except:
                traceback.print_exc()
                logging.info(' '.join(['error_path_2', pic_path]))
    data = np.asarray(data, dtype=np.float32)
    # data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    data = data / 255.0
    label = np.asarray(label)
    return data, label


def load_new_data(folder, shape, part_func=None):
    '''
        每个人一个文件夹,每个人的第一张照片train,其余照片valid
    '''
    person_list = os.listdir(folder)
    person_list.sort()
    train_data = []
    train_label = []
    valid_data = []
    valid_label = []
    for person in person_list:
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        pic_list.sort()
        for index_pic, pic in enumerate(pic_list):
            pic_path = os.path.join(person_path, pic)
            try:
                arr = np.array(Image.open(pic_path))
                arr = imresize(arr, shape)
                if part_func != None:
                    landmarks = get_landmarks(arr)
                    angle = cal_angel(landmarks)
                    rotate_im = rotate(arr, angle)
                    rotate_im = np.array(Image.fromarray(np.uint8(rotate_im*255)))
                    new_landmarks = get_landmarks(rotate_im)
                    arr = part_func(rotate_im, new_landmarks)
                if index_pic < 2:
                    train_data.append(arr)
                    train_label.append(person)
                else:
                    valid_data.append(arr)
                    valid_label.append(person)
            except:
                traceback.print_exc()
                logging.info(' '.join(['error_path_2', pic_path]))
    train_data = np.asarray(train_data, dtype=np.float32)
    train_data = train_data / 255.0
    train_label = np.asarray(train_label)
    valid_data = np.asarray(valid_data, dtype=np.float32)
    valid_data = valid_data / 255.0
    valid_label = np.asarray(valid_label)
    return train_data, train_label, valid_data, valid_label


class Search():
    def __init__(self, model_type, n_estimators=20, n_candidates=200, n_neighbors=10):
        self.lshf = LSHForest(n_estimators=n_estimators, n_candidates=n_candidates, n_neighbors=n_neighbors)

        if model_type == 'rgb_small':
            self.deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
            self.deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'
            self.part_func = None
            self.pic_shape = (50, 50, 3)
            self.feature_dim = 1024
        elif model_type == 'rgb_big':
            self.deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.rgb.deepid.weight'
            self.deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.rgb.deepid.model'
            self.part_func = None
            self.pic_shape = (128, 128, 3)
        elif model_type == 'rgb_small_right':
            self.deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.small.rgb.right_eye.deepid.weight'
            self.deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.small.rgb.right_eye.deepid.model'
            self.part_func = get_right_eye
            self.pic_shape = (50, 50, 3)
        elif model_type == 'rgb_small_left':
            self.deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.small.rgb.left_eye.deepid.weight'
            self.deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.small.rgb.left_eye.deepid.model'
            self.part_func = get_left_eye
            self.pic_shape = (50, 50, 3)
        elif model_type == 'rgb_small_nose':
            self.deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.weight'
            self.deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.model'
            self.part_func = get_nose
            self.pic_shape = (50, 50, 3)
        elif model_type == 'new_shape':
            self.deepid_model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.new_shape.rgb.deepid.model'
            self.deepid_weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.small_data.new_shape.rgb.deepid.weight'
            self.pic_shape = (156, 124, 3)
            self.feature_dim = 256
            self.part_func = None
        self.model, self.get_Conv_FeatureMap = load_deepid_model(self.deepid_model_file, self.deepid_weight_file)
        self.all_label = None
        self.all_feature_data = None


    def extract_pic_feature(self, pic_data, batch_size=128, feature_dim=1024):
        pic_feature = np.zeros(shape=(pic_data.shape[0], feature_dim))
        batch_num = pic_data.shape[0] / batch_size
        for index in range(batch_num):
            # pic_feature[index*batch_size:(index+1)*batch_size, :] = \
            #     self.get_Conv_FeatureMap([pic_data[index*batch_size:(index+1)*batch_size], 0])[0]
            pic_feature[index*batch_size:(index+1)*batch_size, :] = \
                self.get_Conv_FeatureMap([np.transpose(pic_data[index*batch_size:(index+1)*batch_size], (0, 3, 1, 2)), 0])[0]

        if batch_num*batch_size < pic_data.shape[0]:
            # pic_feature[batch_num*batch_size:, :] = \
            #     self.get_Conv_FeatureMap([pic_data[batch_num*batch_size:], 0])[0]
            pic_feature[batch_num*batch_size:, :] = \
                self.get_Conv_FeatureMap([np.transpose(pic_data[batch_num*batch_size:], (0, 3, 1, 2)), 0])[0]
        return pic_feature


    def train_all_data(self, vgg_folder, person_num=100, batch_person_num=20, pic_num=10):
        # 取前pic_num张图片加入到LSH Forest,其余图片用于判断准确率
        for index in range(0+train_person_start_index, person_num+train_person_start_index, batch_person_num):
            if index == 0+train_person_start_index:
                pic_data, all_label = load_batch_train_data(vgg_folder, shape=self.pic_shape, start_person_index=index,
                                 pic_num=pic_num, batch_num=batch_person_num, is_train=True, part_func=self.part_func)
                all_data_feature = self.extract_pic_feature(pic_data, feature_dim=self.feature_dim)
                self.lshf.fit(all_data_feature, all_label)

            else:
                pic_data, this_label = load_batch_train_data(vgg_folder, start_person_index=index, pic_num=pic_num,
                                shape=self.pic_shape, batch_num=batch_person_num,is_train=True, part_func=self.part_func)
                all_label = np.row_stack((np.reshape(all_label, (all_label.shape[0], 1)),
                                          np.reshape(this_label, (this_label.shape[0],1))))
                pic_data_feature = self.extract_pic_feature(pic_data, feature_dim=self.feature_dim)
                all_data_feature = np.row_stack((pic_data_feature, all_data_feature))
                self.lshf.partial_fit(pic_data_feature, this_label)
        self.all_label = all_label
        self.all_feature_data = all_data_feature
        logging.info(' '.join(map(str, ['self.all_label.shape :', self.all_label.shape])))


    def partical_fit(self, pic_data, this_label):
        '''
            增量训练, 样本比较小, 直接
        :param data:
        :param label:
        :return:
        '''
        pic_data_feature = self.extract_pic_feature(pic_data, feature_dim=self.feature_dim)
        self.lshf.partial_fit(pic_data_feature, this_label)
        self.all_label = np.row_stack((np.reshape(self.all_label, (self.all_label.shape[0], 1)),
                                          np.reshape(this_label, (this_label.shape[0],1))))


    def find_k_neighbors(self, pic_data):
        pic_data_feature = self.extract_pic_feature(pic_data, feature_dim=self.feature_dim)
        distances, indices = self.lshf.kneighbors(pic_data_feature, n_neighbors=1)
        predict_label = self.all_label[indices][:, 0, 0]
        return predict_label


    def valid_model(self, vgg_folder, person_num=100, batch_person_num=20, pic_num=10, topK_acc=1):
        # 取前50张图片加入到LSH Forest,后50张图片用于判断准确率
        right_num = 0
        wrong_num = 0
        clf = cPickle.load(open(clf_model_file, 'rb'))

        for index in range(0+train_person_start_index, person_num+train_person_start_index, batch_person_num):
            pic_data, all_label = load_batch_train_data(vgg_folder, start_person_index=index, pic_num=pic_num,
                            shape=self.pic_shape,batch_num=batch_person_num, is_train=False, part_func=self.part_func)

            pic_data_feature = self.extract_pic_feature(pic_data, feature_dim=self.feature_dim)
            distances, indices = self.lshf.kneighbors(pic_data_feature, n_neighbors=10)
            train_data = self.all_feature_data[indices]
            predict_label = self.all_label[indices][:, 0, 0]
            for label_index in range(len(predict_label)):
                this_predict_data = np.abs(train_data[0] - pic_data_feature[0])
                this_result = clf.predict_proba(this_predict_data)
                print this_result
                # pdb.set_trace()
                if all_label[label_index] in self.all_label[indices][:, :, 0][label_index][:topK_acc]:
                    right_num += 1
                else:
                    wrong_num += 1
        acc = right_num * 1.0 / (right_num + wrong_num)
        logging.info(' '.join(map(str, ['model_type :', model_type, 'person_num :', person_num, 'pic_num :', pic_num, 'acc :', acc])))



if __name__ == '__main__':
    model_type = sys.argv[1]
    person_num = int(sys.argv[2])
    topK_acc = int(sys.argv[3])
    vgg_folder = '/data/liubo/face/vgg_face_dataset/all_data/pictures_box'
    small_folder_num = 0
    search = Search(model_type)

    for pic_num in [1, 3]:
        logging.info(' '.join(map(str, ['pic_num :', pic_num, 'person_num :', person_num])))
        search.train_all_data(vgg_folder, person_num=person_num, pic_num=pic_num)
        search.valid_model(vgg_folder, person_num=person_num, pic_num=pic_num, topK_acc=topK_acc)
