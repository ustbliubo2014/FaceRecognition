# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: light_cnn_model.py
@time: 2016/11/14 18:40
@contact: ustb_liubo@qq.com
@annotation: light_cnn_model
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
from sklearn.cross_validation import KFold
import msgpack_numpy
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import sklearn.metrics.pairwise as pw
import pdb
from sklearn.metrics import roc_auc_score
from recog_util import read_one_rgb_pic
import cv2
import shutil
import base64
import keras.backend as K

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


size = 96
pic_shape = (size, size, 3)
FEATURE_DIM = 512
PIC_SHAPE = (3, size, size)
upper_verif_threshold = 0.55
lower_verif_threshold = 0.35
port = 6666
nearest_num = 5   # 计算当前图片与以前5张图片的相似度
same_pic_threshold = 0.9
pair_file = '/home/liubo-it/FaceRecognization/CNN_Model/self_all_pair.txt'
feature_pack_file = 'light_cnn_feature_pack.p'


def train_valid_verif_model():
    all_data = []
    all_label = []
    all_pic_path_list = []
    count = 0
    for line in open(pair_file):
        if count % 100 == 0:
            print count
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            path1 = tmp[0]
            path2 = tmp[1]
            if (os.path.exists(path1)) and (os.path.exists(path2)):
                feature1 = extract_feature_from_file(path1)
                feature2 = extract_feature_from_file(path2)
                predicts = pw.cosine_similarity(feature1, feature2)
                all_data.append(predicts)
                all_label.append(int(tmp[2]))
    msgpack_numpy.dump((all_data, all_label, all_pic_path_list), open(feature_pack_file, 'wb'))
    (all_data, all_label, all_pic_path_list) = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    all_data = np.asarray(all_data)
    data = np.reshape(all_data, newshape=(all_data.shape[0], all_data.shape[2]))
    label = np.asarray(all_label)
    print data.shape, label.shape

    kf = KFold(len(label), n_folds=10)
    all_acc = []
    for (train, valid) in kf:
        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]
        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        roc_auc = roc_auc_score(valid_label, clf.predict(valid_data))
        all_acc.append(acc)
        print acc, roc_auc
    print np.mean(all_acc)

    clf = LinearSVC()
    clf.fit(data, label)
    pdb.set_trace()
    cPickle.dump(clf, open(verification_model_file, 'wb'))


def load_model():
    model_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_local_10575.model'
    weight_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_local_10575.weight'
    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load weights'
        model.load_weights(weight_file)
        get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                         [model.layers[-3].get_output_at(False)])
        return model, get_Conv_FeatureMap


model, get_Conv_FeatureMap = load_model()
verification_model_file = '/data/liubo/face/vgg_face_model/self_verification_light_cnn_model'
verification_model = cPickle.load(open(verification_model_file, 'rb'))
verification_same_person = 0
face_tmp_file = 'add_face.png'


def extract_feature_from_file(pic_path):
    # 包含人脸检测和人脸识别
    # 训练数据使用dlib+align, 所以提取特征时也可以采用这种方法
    # 将人脸写回到原来的位置, 下次加载的时候不会进行人脸检测,可以和原来保持一致
    # 人脸识别的时候不用, 添加新图片的时候会用到(用户放入的图片可能不是对齐的图片)
    new_face = align_face(pic_path)
    if new_face != None:
        cv2.imwrite(pic_path, new_face)
    img = read_one_rgb_pic(pic_path, pic_shape)
    if K.image_dim_ordering() != 'th':
        img = np.transpose(img, (0, 2, 3, 1))
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


def get_all_img_feature():
    feature_result_file = '/tmp/annotate_light_cnn_feature.p'
    f = open(feature_result_file, 'w')
    folder = '/tmp/all_images'
    person_list = os.listdir(folder)
    for person in person_list:
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            feature = np.asarray(extract_feature_from_file(pic_path)[0])
            f.write(base64.b64encode(msgpack_numpy.dumps((feature, person_path)))+'\n')
    f.close()


if __name__ == '__main__':
    pass


    # train_valid_verif_model()

    # get_all_img_feature()

    pic_path1 = sys.argv[1]
    feature1 = extract_feature_from_file(pic_path1)
    pic_path2 = sys.argv[2]
    feature2 = extract_feature_from_file(pic_path2)
    print pw.cosine_similarity(feature1, feature2)
