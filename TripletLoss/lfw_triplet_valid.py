# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: lfw_vgg_valid.py
@time: 2016/7/26 10:47
@contact: ustb_liubo@qq.com
@annotation: lfw_triplet_valid
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='lfw_vgg_valid.log',
                    filemode='a+')

import numpy as np
# import _init_paths
# import caffe
from scipy.misc import imread, imsave, imresize
import pdb
import msgpack_numpy
from sklearn.cross_validation import train_test_split
import msgpack_numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import cPickle
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.spatial.distance import euclidean, cosine
import sklearn.metrics.pairwise as pw
from time import time
import traceback
from keras.models import model_from_json
import heapq
from keras.optimizers import Adam
import keras.backend as K
from sklearn.model_selection import KFold

avg = np.array([129.1863, 104.7624, 93.5940])
lfw_folder = '/data/liubo/face/lfw_face'
pair_file = '/data/liubo/face/lfw_pair.txt'
feature_pack_file = '/data/liubo/face/lfw_vgg_feature.p'
triplet_feature_pack_file = '/data/liubo/face/triplet_lfw_vgg_feature.p'
pair_feature_pack_file = '/data/liubo/face/lfw_pair_feature.p'
pair_dist_pack_file = '/data/liubo/face/lfw_pair_dist.p'
error_pair_file = '/data/liubo/face/error_pair.txt'
new_pair_pack_file = '/data/liubo/face/lfw_new_pair.p'
pic_shape = (224, 224, 3)
layer = 'fc7'


# caffe.set_mode_gpu()
# caffe.set_device(3)
# net = caffe.Net("/home/liubo-it/VGGFaceModel-master/VGG_FACE_deploy.prototxt",
#                 "/home/liubo-it/VGGFaceModel-master/VGG_FACE.caffemodel",
#                 caffe.TEST)


def read_one_rgb_pic(pic_path, pic_shape=(224, 224, 3)):
    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def extract(pic_path):
    img = read_one_rgb_pic(pic_path, pic_shape)
    start = time()
    net.blobs['data'].data[...] = img
    net.forward(data=img)
    conv_feature = net.blobs['fc7'].data[0].copy()
    end = time()
    # print 'extract feature time :', (end - start)
    return np.reshape(conv_feature, conv_feature.size)


def filter_path(this_person_feature_list, index_list):
    # 一张图片里有两个人时,这两个人的图片都删掉
    try:
        tmp_list = []
        for index, e in enumerate(this_person_feature_list):
            feature, path = e
        # Alexandre_Despatie_0001.jpg_face_0.jpg
        # Alexandre_Despatie_0001.jpg_face_1.jpg
        # Alina_Kabaeva_0001.jpg_face_0.jpg
            if 'face_0' not in path:
                tmp_list.append(path[:-5])
        for index, e in enumerate(this_person_feature_list):
            feature, path = e
            for del_path in tmp_list:
                if del_path in path:
                    index_list.remove(index)
                    break
    except:
        traceback.print_exc()


def extract_lfw_feature():
    lfw_feature_dic = {} # {person:[feature1,feature2,...,]}
    person_list = os.listdir(lfw_folder)
    for person_index, person in enumerate(person_list):
        print person_index, person
        person_path = os.path.join(lfw_folder, person)
        pic_list = os.listdir(person_path)
        this_person_feature_list = []
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            this_feature = extract(pic_path)
            this_person_feature_list.append((this_feature, os.path.join(person, pic)))
        lfw_feature_dic[person] = this_person_feature_list
    msgpack_numpy.dump(lfw_feature_dic, open(feature_pack_file, 'wb'))


def main_distance():
    # lfw_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    lfw_feature_dic = msgpack_numpy.load(open(triplet_feature_pack_file, 'rb'))
    data = []
    label = []
    pic_path_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0] #取该人的两个特征向量
            this_person_feature_list = lfw_feature_dic.get(person, [])
            index_list = range(len(this_person_feature_list))
            np.random.shuffle(index_list)
            filter_path(this_person_feature_list, index_list)
            if len(index_list) < 2:
                continue
            feature1, path1 = this_person_feature_list[index_list[0]]
            feature2, path2 = this_person_feature_list[index_list[1]]
            feature1 = np.reshape(feature1, newshape=(1, feature1.size))
            feature2 = np.reshape(feature2, newshape=(1, feature2.size))
            predicts = pw.cosine_similarity(feature1, feature2)
            label.append(0)
            data.append(predicts)
            pic_path_list.append('\t'.join([path1, path2]))
        elif len(tmp) == 4:
            person1 = tmp[0]
            person2 = tmp[2]
            # 每个人分别取一个特征向量
            this_person_feature_list1 = lfw_feature_dic.get(person1, [])
            this_person_feature_list2 = lfw_feature_dic.get(person2, [])
            index_list1 = range(len(this_person_feature_list1))
            index_list2 = range(len(this_person_feature_list2))
            np.random.shuffle(index_list1)
            np.random.shuffle(index_list2)
            filter_path(this_person_feature_list1, index_list1)
            filter_path(this_person_feature_list2, index_list2)
            if len(index_list1) < 1 or len(index_list2) < 1:
                continue
            index_list1 = np.arange(len(this_person_feature_list1))
            index_list2 = np.arange(len(this_person_feature_list2))
            np.random.shuffle(index_list1)
            np.random.shuffle(index_list2)
            feature1, path1 = this_person_feature_list1[index_list1[0]]
            feature2, path2 = this_person_feature_list2[index_list2[0]]
            feature1 = np.reshape(feature1, newshape=(1, feature1.size))
            feature2 = np.reshape(feature2, newshape=(1, feature2.size))
            predicts = pw.cosine_similarity(feature1, feature2)
            label.append(1)
            data.append(predicts)
            pic_path_list.append('\t'.join([path1, path2]))
    data = np.asarray(data)
    print data.shape
    data = np.reshape(data, newshape=(len(data), 1))
    label = np.asarray(label)
    print data.shape, label.shape

    kf = KFold(n_folds=10)
    all_acc = []
    for k, (train, valid) in enumerate(kf.split(data, label)):

        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        all_acc.append(acc)
        print 'acc :', acc

    print 'mean acc :', np.mean(all_acc)


def verfi_two_pic(pic_path1, pic_path2):
    feature1 = extract(pic_path1)
    feature2 = extract(pic_path2)
    clf = cPickle.load(open('/data/liubo/face/vgg_face_dataset/model/verification_model', 'rb'))
    predicts = pw.cosine_similarity(feature1, feature2)
    result = clf.predict(predicts)
    print result


def find_max_min():
    # 同一个人里找相似度最小的, 不同人里找相似度最大的
    lfw_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    person_list = lfw_feature_dic.keys()
    same_person_score = []
    same_person_score_pair_dic = {}   # {score:[(path1,path2), ...,(path1,path2)]}
    no_same_person_score = []
    no_same_person_score_pair_dic = {}   # {score:[(path1,path2), ...,(path1,path2)]}
    heapq.heapify(same_person_score)
    pair_threshold = 3000

    for person_index, person in enumerate(person_list):
        start = time()
        path_feature_list = lfw_feature_dic.get(person)
        # 找出该人里所有可能的pair --- score越小越好(同一个人最不相似的照片)
        # 每次将最大的score去掉,加入更小的score,所以在加入是score取负,这样堆顶就是原来score最大的值
        length = len(path_feature_list)
        for index_i in range(length):
            for index_j in range(index_i, length):
                feature1, path1 = path_feature_list[index_i]
                feature2, path2 = path_feature_list[index_j]
                feature1 = np.reshape(feature1, newshape=(1, feature1.shape[0]))
                feature2 = np.reshape(feature2, newshape=(1, feature2.shape[0]))
                this_score = 0 - pw.cosine_similarity(feature1, feature2)[0][0]
                if len(same_person_score) > pair_threshold:
                    top_item = same_person_score[0]
                    if this_score < top_item:    # 更加不相似,加入
                        heapq.heappop(same_person_score)
                        heapq.heappush(same_person_score, this_score)
                        # 删除原来的pair, 加入当前pair (同一个分数可能对应于多个pair)
                        if top_item in same_person_score_pair_dic:
                            same_person_score_pair_dic.pop(top_item)
                        pair_list = same_person_score_pair_dic.get(this_score, [])
                        pair_list.append((path1, path2))
                        same_person_score_pair_dic[this_score] = pair_list
                else:
                    heapq.heappush(same_person_score, this_score)
                    pair_list = same_person_score_pair_dic.get(this_score, [])
                    pair_list.append((path1, path2))
                    same_person_score_pair_dic[this_score] = pair_list

        # 找出所有可能的不相似的pair

        for other_person_index, other_person in enumerate(person_list[person_index+1:], start=person_index+1):
            other_path_feature_list = lfw_feature_dic.get(other_person)
            if other_person == person:
                continue
            other_length = len(other_path_feature_list)
            for index_i in range(length):
                for index_j in range(other_length):
                    feature1, path1 = path_feature_list[index_i]
                    feature2, path2 = other_path_feature_list[index_j]
                    feature1 = np.reshape(feature1, newshape=(1, feature1.shape[0]))
                    feature2 = np.reshape(feature2, newshape=(1, feature2.shape[0]))
                    this_score = pw.cosine_similarity(feature1, feature2)[0][0]
                    if len(no_same_person_score) > pair_threshold:
                        top_item = no_same_person_score[0]
                        if this_score < top_item:    # 更加相似, 加入
                            heapq.heappop(no_same_person_score)
                            heapq.heappush(no_same_person_score, this_score)
                            # 删除原来的pair, 加入当前pair (同一个分数可能对应于多个pair)
                            if top_item in no_same_person_score_pair_dic:
                                no_same_person_score_pair_dic.pop(top_item)
                            pair_list = no_same_person_score_pair_dic.get(this_score, [])
                            pair_list.append((path1, path2))
                            no_same_person_score_pair_dic[this_score] = pair_list
                    else:
                        heapq.heappush(no_same_person_score, this_score)
                        pair_list = no_same_person_score_pair_dic.get(this_score, [])
                        pair_list.append((path1, path2))
                        no_same_person_score_pair_dic[this_score] = pair_list
        end = time()
        print person_index, person, (end - start), length
    msgpack_numpy.dump((same_person_score_pair_dic, same_person_score, no_same_person_score_pair_dic, no_same_person_score),
                       open(new_pair_pack_file, 'wb'))


def extract_triplet_feature():
    lfw_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    new_lfw_feature_dic = {}
    model_file = '/data/liubo/face/vgg_face_model/annotate_siamese_graph.model'
    weight_file = '/data/liubo/face/vgg_face_model/annotate_siamese_graph.weight'
    model = model_from_json(open(model_file, 'r').read())
    opt = Adam()
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    model.load_weights(weight_file)

    # pdb.set_trace()
    get_Conv_FeatureMap = K.function([model.layers[2].layers[0].get_input_at(False), K.learning_phase()],
                                     [model.layers[2].layers[-1].get_output_at(False)])
    for person in lfw_feature_dic:
        # print person
        this_person_feature_list = lfw_feature_dic.get(person)
        this_person_triplet_feature_list = []
        for feature, path in this_person_feature_list:
            feature = np.reshape(feature, (1, feature.size))
            new_feature = get_Conv_FeatureMap([feature, 0])[0].copy()
            this_person_triplet_feature_list.append((new_feature, path))

        new_lfw_feature_dic[person] = this_person_triplet_feature_list
    msgpack_numpy.dump(new_lfw_feature_dic, open(triplet_feature_pack_file, 'wb'))


if __name__ == '__main__':
    # verfi_two_pic(sys.argv[1], sys.argv[2])

    # extract_lfw_feature()
    # main_distance()
    #
    extract_triplet_feature()
    main_distance()