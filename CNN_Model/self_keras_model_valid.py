# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: self_keras_model_valid.py
@time: 2016/11/14 17:20
@contact: ustb_liubo@qq.com
@annotation: self_keras_model_valid
"""
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
from scipy.misc import imread, imresize
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
pair_file = 'self_all_pair.txt'
error_pair_file = '/data/liubo/face/error_dlib_pair.txt'
pic_shape = (96, 96, 3)


def read_one_rgb_pic(pic_path, pic_shape=pic_shape):
    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


def load_model():
    # model_file = '/data/liubo/face/annotate_face_model/light_cnn_10000.model'
    # weight_file = '/data/liubo/face/annotate_face_model/light_cnn_10000.weight'

    # model_file = '/data/liubo/face/annotate_face_model/light_cnn_10575.model'
    # weight_file = '/data/liubo/face/annotate_face_model/light_cnn_10575.weight'
    #
    model_file = '/data/liubo/face/annotate_face_model/light_cnn_61962.model'
    weight_file = '/data/liubo/face/annotate_face_model/light_cnn_61962.weight'

    # model_file = '/data/liubo/face/annotate_face_model/center_loss_10575.model'
    # weight_file = '/data/liubo/face/annotate_face_model/center_loss_10575.weight'

    # model_file = '/data/liubo/face/annotate_face_model/light_cnn_5821.model'
    # weight_file = '/data/liubo/face/annotate_face_model/light_cnn_5821.weight'

    if os.path.exists(model_file) and os.path.exists(weight_file):
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
        print 'load weights'
        model.load_weights(weight_file)
        return model


model = load_model()
get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                 [model.layers[-3].get_output_at(False)])


def extract(pic_path):
    start = time()
    img = read_one_rgb_pic(pic_path, pic_shape)
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    end = time()
    print 'extract feature time :', (end - start)
    return feature_vector


def main_distance():
    data = []
    label = []
    pic_path_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        path1 = tmp[0]
        path2 = tmp[1]
        if (not os.path.exists(path1)) or (not os.path.exists(path2)):
            continue
        label.append(int(tmp[2]))
        feature1 = extract(path1)
        feature2 = extract(path2)
        predicts = pw.cosine_similarity(feature1, feature2)
        data.append(predicts)
        pic_path_list.append('\t'.join([path1, path2]))

    data = np.asarray(data)
    data = np.reshape(data, newshape=(data.shape[0], 1))
    label = np.asarray(label)
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


if __name__ == '__main__':
    main_distance()
