# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_gray.py
@time: 2016/7/1 20:38
@contact: ustb_liubo@qq.com
@annotation: DeepId_gray
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
import os
import msgpack_numpy
from DeepID.DeepId1.DeepId import train_valid_deepid, extract_feature
import pdb
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from DeepID.data_process.vgg_face.train_valid_test_split import pack_data
from DeepID.data_process.vgg_face.load_data import load_test_data,load_gray_test_data
from time import time
from Queue import Queue
import threading
import cPickle
import msgpack
from sklearn.neighbors import KDTree
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from PIL import Image
from scipy.misc import imread, imsave, imresize
from CriticalPointDetection.split_pic import get_landmarks, get_nose, get_left_eye, get_right_eye, cal_angel
import traceback
from skimage.transform import rotate

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='DeepId_gray.log',
                    filemode='w')


def load_train_data(folder, shape=(128, 128), need_pic_list=False, pic_num_threshold=1000, part_func=None):
    person_list = os.listdir(folder)
    data = []
    label = []
    all_pic_list = []
    label_trans_dic = {}
    current_label = 0
    for person_index, person in enumerate(person_list):
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
                arr = np.array(Image.open(pic_path).convert('L'))
                arr = imresize(arr, shape)
                if part_func != None:
                        landmarks = get_landmarks(arr)
                        angle = cal_angel(landmarks)
                        rotate_im = rotate(arr, angle)
                        rotate_im = np.array(Image.fromarray(np.uint8(rotate_im*255)))
                        new_landmarks = get_landmarks(rotate_im)
                        arr = part_func(rotate_im, new_landmarks)
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



def train_vgg_face():
    data_folder = '/data/liubo/face/vgg_face_dataset/'
    data, label = load_train_data('/data/liubo/face/vgg_face_dataset/all_data/pictures_box',
                                  shape=(128, 128), pic_num_threshold=50, part_func=None)
    X_train, X_test, y_train,  y_test = train_test_split(data, label, test_size=0.2)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    # nb_classes = len(set(list(y_train)))
    nb_classes = 2622
    # pdb.set_trace()

    print 'all shape : ',X_train.shape, y_train.shape, X_test.shape, y_test.shape

    weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.bn.deepid.weight'
    model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.bn.deepid.model'
    print 'model_file : ', model_file, ' weight_file : ', weight_file
    error_valid_sample_file = os.path.join(data_folder,'error_valid_sample.txt')
    error_train_sample_file = os.path.join(data_folder,'error_train_sample.txt')
    train_valid_deepid(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file,
                       error_train_sample_file, error_valid_sample_file)


def extract_vgg_face_feature():
    test_data, test_label, label_trans_dic = pack_data('/data/liubo/face/self', '/data/liubo/face/self.p', is_test=True)
    print 'all shape : ',test_data.shape, test_label.shape
    X_test = np.transpose(test_data, (0,3,1,2))

    weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.train_valid.small.rgb.deepid.weight.bak'
    model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.train_valid.small.rgb.deepid.model.bak'
    feature_dim = 1024
    print 'model_file : ', model_file, ' weight_file : ', weight_file
    X_test_feature = extract_feature(X_test, model_file, weight_file, feature_dim)
    print 'X_test_feature.shape', X_test_feature.shape

    tree = KDTree(X_test_feature)
    cPickle.dump(tree, open('/data/liubo/face/vgg_face_dataset/model/tree_rotate.model','wb'))
    msgpack.dump(label_trans_dic, open('/data/liubo/face/vgg_face_dataset/model/label_trans_dic_rotate.p', 'wb'))
    msgpack_numpy.dump(test_label, open('/data/liubo/face/vgg_face_dataset/model/y_train_rotate.p','wb'))
    valid_acc_list = []
    for index in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X_test_feature, test_label, test_size=0.7, random_state=0)
        tree = KDTree(X_train)
        right_num = 1
        wrong_num = 1
        clf = NearestCentroid()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        print 'NearestCentroid :', accuracy_score(y_test, y_predict)
        start = time()
        for index in range(len(y_test)):
            clf.predict(X_test[index:index+1,:])
        end = time()
        print 'signal test time :', (end - start)
        for same_person_num in range(1, 11):
            start = time()
            for index in range(X_test.shape[0]):
                dist, ind = tree.query(X_test[index:index+1,:], k=10)
                counter = Counter(y_train[ind][0]).items()
                counter.sort(key=lambda x:x[1], reverse=True)
                if counter[0][1] >= same_person_num:
                    this_label = counter[0][0]
                    if this_label == y_test[index]:
                        right_num += 1
                    else:
                        wrong_num += 1
            end = time()
            acc = right_num*1.0/(right_num+wrong_num)
            print 'time :', (end - start), 'same_person_num :', same_person_num, \
                'right num :', (right_num), 'wrong_num :', (wrong_num) , 'acc :', acc
            valid_acc_list.append(acc)
        print 'test mean acc :', np.mean(valid_acc_list)



if __name__ == '__main__':
    train_vgg_face()
    # extract_vgg_face_feature()

