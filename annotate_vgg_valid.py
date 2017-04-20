# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: lfw_vgg_valid.py
@time: 2016/7/26 10:47
@contact: ustb_liubo@qq.com
@annotation: lfw_vgg_valid
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
import numpy as np
from scipy.misc import imread, imsave, imresize
import pdb
from sklearn.cross_validation import train_test_split
import msgpack_numpy
import cPickle
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import sklearn.metrics.pairwise as pw
import traceback
from keras.models import model_from_json
import keras.backend as K
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

avg = np.array([129.1863, 104.7624, 93.5940])
# 测试标注数据
pair_file = '/data/verif_list.txt'
feature_pack_file = '/data/liubo/face/annotate_vgg_feature.p'
error_pair_file = '/data/liubo/face/error_pair.txt'
pic_shape = (224, 224, 3)


def read_one_rgb_pic(pic_path, pic_shape=pic_shape):
    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def load_model():
    model_file = '/data/liubo/face/vgg_face_dataset/model/annotate_deep_face_757.model'
    weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate_deep_face_757.weight'
    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load_weights'
        model.load_weights(weight_file)
        return model


model = load_model()
# pdb.set_trace()
get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                 [model.layers[-5].get_output_at(False)])


def extract(pic_path):
    img = read_one_rgb_pic(pic_path, pic_shape)
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    return feature_vector

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


def main_distance():
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
            label = int(tmp[2])
            feature1 = extract(path1)
            feature2 = extract(path2)
            # print feature1.shape, feature2.shape
            # feature1 = np.reshape(feature1, newshape=(1, feature1.shape[0]))
            # feature2 = np.reshape(feature2, newshape=(1, feature2.shape[0]))
            predicts = pw.cosine_similarity(feature1, feature2)
            # predicts = np.fabs(feature1 - feature2)
            all_data.append(predicts)
            all_label.append(label)
            all_pic_path_list.append((path1, path2))
    msgpack_numpy.dump((all_data, all_label, all_pic_path_list), open(feature_pack_file, 'wb'))


def verfi_two_pic(pic_path1, pic_path2):
    feature1 = extract(pic_path1)
    feature2 = extract(pic_path2)
    clf = cPickle.load(open('/data/liubo/face/vgg_face_dataset/model/verification_model', 'rb'))
    predicts = pw.cosine_similarity(feature1, feature2)
    result = clf.predict(predicts)
    print result


if __name__ == '__main__':
    # verfi_two_pic(sys.argv[1], sys.argv[2])

    main_distance()

    (all_data, all_label, all_pic_path_list) = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    all_data = np.asarray(all_data)
    all_data = np.reshape(all_data, newshape=(all_data.shape[0], all_data.shape[2]))
    all_label = np.asarray(all_label)
    all_pic_path_list = np.asarray(all_pic_path_list)
    print all_data.shape, all_label.shape

    all_acc = []
    f = open(error_pair_file, 'w')
    kf = KFold(n_folds=10)
    all_acc = []
    for k, (train, valid) in enumerate(kf.split(all_data, all_label, all_pic_path_list)):
        train_data = all_data[train]
        valid_data = all_data[valid]
        train_label = all_label[train]
        valid_label = all_label[valid]
        train_path_list = all_pic_path_list[train]
        valid_path_list = all_pic_path_list[valid]

        clf = LinearSVC()
        # clf = RandomForestClassifier(n_estimators=500)
        # clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=500)
        # clf = SVC()
        clf.fit(train_data, train_label)
        predict_label = clf.predict(valid_data)
        for index in range(len(predict_label)):
            if valid_label[index] != predict_label[index]:
                f.write(valid_path_list[index][0]+'\t'+valid_path_list[index][1]+'\t'+
                        str(valid_label[index])+'\t'+str(predict_label[index])+'\n')
        acc = accuracy_score(valid_label, clf.predict(valid_data))
        all_acc.append(acc)
        print acc
    f.close()
    print 'mean_acc :', np.mean(all_acc)

