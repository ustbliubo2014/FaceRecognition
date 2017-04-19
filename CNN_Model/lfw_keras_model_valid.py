# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: lfw_keras_vgg_valid.py
@time: 2016/8/8 12:39
@contact: ustb_liubo@qq.com
@annotation: lfw_keras_model_valid
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
from scipy.misc import imread, imresize
from sklearn.cross_validation import train_test_split
import msgpack_numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import cPickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import sklearn.metrics.pairwise as pw
import traceback
from keras.models import model_from_json
from keras.optimizers import Adam
import keras.backend as K
import pdb
from sklearn.cross_validation import KFold
from time import time


avg = np.array([129.1863, 104.7624, 93.5940])
lfw_folder = '/data/liubo/face/lfw/lfw_face_align'
pair_file= '/data/liubo/face/lfw/pairs.txt'
feature_pack_file = '/data/liubo/face/lfw_feature.p'
error_pair_file = '/data/liubo/face/error_pair.txt'
pic_size = 96
pic_shape = (pic_size, pic_size, 3)


def read_one_rgb_pic(pic_path, pic_shape=pic_shape):
    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def load_model():
    # model_file = '/data/liubo/face/annotate_face_model/light_cnn_61962.model'
    # weight_file = '/data/liubo/face/annotate_face_model/light_cnn_61962.weight'
    # mean_acc: 0.960719550317  ---  theano
    # mean_acc: 0.725701339299  ---  tensorflow
    # 不同backend训练的模型不能通用

    # model_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_10575.model'
    # weight_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_10575.weight'
    # mean_acc: 0.944297513551

    # model_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_local_10575.model'
    # weight_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_local_10575.weight'
    # mean_acc: 0.965546027452

    # model_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_local_10575_augment.model'
    # weight_file = '/data/liubo/face/annotate_face_model/thin_casia_dlib_light_cnn_local_10575_augment.weight'
    # mean_acc: 0.962660355098

    # model_file = '/data/liubo/face/annotate_face_model/thin_ms_dlib_light_cnn_tf_79078.model'
    # weight_file = '/data/liubo/face/annotate_face_model/thin_ms_dlib_light_cnn_tf_79078.weight'
    # mean_acc: 0.966566141982

    model_file = '/data/liubo/face/annotate_face_model/thin_ms_dlib_light_cnn_tf_augment_79078.model'
    weight_file = '/data/liubo/face/annotate_face_model/thin_ms_dlib_light_cnn_tf_augment_79078.weight'
    # mean_acc: 0.972336911168

    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load weights'
        model.load_weights(weight_file)
        return model


model = load_model()
model.summary()
#pdb.set_trace()
get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                 [model.layers[-3].get_output_at(False)])


def extract(pic_path):
    img = read_one_rgb_pic(pic_path, pic_shape)
    start = time()
    if K.image_dim_ordering() != 'th':
        img = np.transpose(img, (0, 2, 3, 1))
    # 将theano的模型用tensorflow处理, 还是使用theano的dim
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    end = time()
    # print 'time :', end - start
    return feature_vector


def extract_lfw_feature():
    lfw_feature_dic = {} # {person:[feature1,feature2,...,]}
    person_list = os.listdir(lfw_folder)
    for person_index, person in enumerate(person_list):
        print person_index, person
        person_path = os.path.join(lfw_folder, person)
        pic_list = os.listdir(person_path)
        this_person_feature_dic = {}
        for pic in pic_list:
            try:
                pic_path = os.path.join(person_path, pic)
                index = int(pic.split('.')[0].split('_')[-1])
                this_feature = extract(pic_path)
                this_person_feature_dic[index] = (this_feature, pic_path)
            except:
                traceback.print_exc()
                pdb.set_trace()
        lfw_feature_dic[person] = this_person_feature_dic
    msgpack_numpy.dump(lfw_feature_dic, open(feature_pack_file, 'wb'))


def main_distance():
    lfw_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    data = []
    label = []
    pic_path_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0]         # 取该人的两个特征向量
            index1 = int(tmp[1])
            index2= int(tmp[2])
            this_person_feature_dic = lfw_feature_dic.get(person, {})
            if index1 in this_person_feature_dic and index2 in this_person_feature_dic:
                feature1, path1 = this_person_feature_dic[index1]
                feature2, path2 = this_person_feature_dic[index2]
                predicts = pw.cosine_similarity(feature1, feature2)
                label.append(0)
                data.append(predicts)
                pic_path_list.append('\t'.join([path1, path2]))
        elif len(tmp) == 4:
            person1 = tmp[0]
            index1 = int(tmp[1])
            person2 = tmp[2]
            index2 = int(tmp[3])
            # 每个人分别取一个特征向量
            this_person_feature_dic1 = lfw_feature_dic.get(person1, {})
            this_person_feature_dic2 = lfw_feature_dic.get(person2, {})
            if index1 in this_person_feature_dic1 and index2 in this_person_feature_dic2:
                feature1, path1 = this_person_feature_dic1[index1]
                feature2, path2 = this_person_feature_dic2[index2]
                predicts = pw.cosine_similarity(feature1, feature2)
                label.append(1)
                data.append(predicts)
                pic_path_list.append('\t'.join([path1, path2]))
    data = np.asarray(data)
    # data = np.reshape(data, newshape=(data.shape[0], data.shape[-1]))
    data = np.reshape(data, newshape=(data.shape[0], 1))
    label = np.asarray(label)

    pic_path_list = np.asarray(pic_path_list)

    kf = KFold(len(label), n_folds=10)
    all_acc = []
    f = open('error.txt', 'w')
    for (train, valid) in kf:
        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]
        train_path = pic_path_list[train]
        valid_path = pic_path_list[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        roc_auc = roc_auc_score(valid_label, clf.predict(valid_data))
        for index in range(len(valid_data)):
            if valid_label[index] != clf.predict(np.reshape(valid_data[index], (1, 1))):
                f.write(str(index)+'\t'+valid_path[index]+'\n')
        all_acc.append(acc)
        print acc, roc_auc
    f.close()
    all_acc.sort(reverse=True)
    print 'mean_acc :', np.mean(all_acc[:])


if __name__ == '__main__':
    extract_lfw_feature()
    main_distance()
