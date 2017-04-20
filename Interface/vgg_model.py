# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: vgg_model.py
@time: 2016/11/4 18:26
@contact: ustb_liubo@qq.com
@annotation: vgg_model : 提供vgg模型相关的参数和方法
"""

import sys
sys.path.insert(0, '/home/liubo-it/FaceRecognization/')
import logging
from logging.config import fileConfig
import os
from keras.models import model_from_json
from keras.optimizers import Adam
import keras.backend as K
from recog_util import read_one_rgb_pic
import traceback
import cPickle
from DetectAndAlign.align_interface import align_face
from sklearn.model_selection import KFold
import msgpack_numpy
import msgpack
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.metrics.pairwise as pw
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


pic_shape = (224, 224, 3)
FEATURE_DIM = 4096
PIC_SHAPE = (3, 224, 224)
upper_verif_threshold = 0.55
lower_verif_threshold = 0.35
port = 6666
nearest_num = 5   # 计算当前图片与以前5张图片的相似度
nearest_time_threshold = 30
pair_file = 'self_all_pair.txt'
feature_pack_file = 'vgg_feature_pack.p'


def create_train_valid_data(folder='/data/liubo/face/crop_face'):
    # 根据已经存在的数据训练人脸验证模型
    person_list = os.listdir(folder)
    path_feature_dic = {}  #
    for person in person_list:
        print person
        person_path = os.path.join(folder, person)
        pic_feature_list = os.listdir(person_path)
        for pic_feature_path in pic_feature_list:
            pic_feature_path = os.path.join(person_path, pic_feature_path)
            pic_feature = extract_feature_from_file(pic_feature_path)
            path_feature_dic[pic_feature_path] = pic_feature
    msgpack_numpy.dump(path_feature_dic, open(feature_pack_file, 'wb'))


def train_valid_verif_model():
    all_data = []
    all_label = []
    all_pic_path_list = []
    count = 0
    path_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    not_in = 0
    not_in_pair = {}
    for line in open(pair_file):
        if count % 100 == 0:
            print count
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            path1 = tmp[0]
            path2 = tmp[1]
            label = int(tmp[2])
            if path1 in path_feature_dic and path2 in path_feature_dic:
                try:
                    feature1 = np.asarray(path_feature_dic.get(path1))
                    feature2 = np.asarray(path_feature_dic.get(path2))
                    predicts = pw.cosine_similarity(feature1, feature2)
                    all_data.append(predicts)
                    all_label.append(label)
                    all_pic_path_list.append((path1, path2))
                except:
                    traceback.print_exc()
            else:
                traceback.print_exc()
    msgpack_numpy.dump((all_data, all_label, all_pic_path_list), open(feature_pack_file, 'wb'))

    (all_data, all_label, all_pic_path_list) = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    pdb.set_trace()
    all_data = np.asarray(all_data)
    all_data = np.reshape(all_data, newshape=(all_data.shape[0], all_data.shape[2]))
    all_label = np.asarray(all_label)
    all_pic_path_list = np.asarray(all_pic_path_list)
    print all_data.shape, all_label.shape

    all_acc = []

    kf = KFold(n_folds=10)
    all_acc = []
    f = open('research_verif_result.txt', 'w')
    for k, (train, valid) in enumerate(kf.split(all_data, all_label, all_pic_path_list)):
        train_data = all_data[train]
        valid_data = all_data[valid]
        train_label = all_label[train]
        valid_label = all_label[valid]
        train_path_list = all_pic_path_list[train]
        valid_path_list = all_pic_path_list[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        for k in range(len(valid_path_list)):
            f.write(os.path.split(valid_path_list[k][0])[1] + '\t' + os.path.split(valid_path_list[k][1])[1] +
                    '\t' + str(valid_data[k][0])+ '\t' + str(valid_label[k]) + '\n')
        all_acc.append(acc)
        print acc
    print 'mean_acc :', np.mean(all_acc)
    f.close()
    clf = LinearSVC()
    clf.fit(all_data, all_label)
    pdb.set_trace()
    cPickle.dump(clf, open(verification_model_file, 'wb'))


def load_model():
    model_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/DeepFace.weight'
    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load weights'
        model.load_weights(weight_file)
        get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                         [model.layers[22].get_output_at(False)])
        return model, get_Conv_FeatureMap

model, get_Conv_FeatureMap = load_model()
verification_model_file = '/data/liubo/face/vgg_face_model/self_verification_model'
verification_model = cPickle.load(open(verification_model_file, 'rb'))
verification_same_person = 0
face_tmp_file = 'add_face.png'


def extract_feature_from_file(pic_path):
    # new_face = align_face(pic_path)
    # # 将人脸写回到原来的位置, 下次加载的时候不会进行人脸检测,可以和原来保持一致
    # if new_face != None:
    #     imsave(pic_path, new_face)
    img = read_one_rgb_pic(pic_path, pic_shape)
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    return feature_vector


def extract_feature_from_numpy(img):
    try:
        if img.shape[1:] == PIC_SHAPE:
            feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
            return feature_vector
        else:
            print 'error shape'
            return None
    except:
        traceback.print_exc()
        return None


if __name__ == '__main__':
    create_train_valid_data()
    train_valid_verif_model()
