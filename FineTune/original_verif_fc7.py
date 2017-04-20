# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: original_verif.py
@time: 2016/8/18 15:11
@contact: ustb_liubo@qq.com
@annotation: original_verif
"""
import sys
import logging
from logging.config import fileConfig
import os
from original_fine_tune_fc7 import load_model
import msgpack_numpy
from conf import *
import numpy as np
from sklearn.cross_validation import train_test_split
import sklearn.metrics.pairwise as pw
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from extract_feature import extract
import pdb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def extract_verif_feature():
    verif_path_feature_dic = {} # {person:[feature1,feature2,...,]}
    # path_feature_dic = msgpack_numpy.load(open(originalimages_verif_fc7_path_feature, 'rb'))
    path_set = set()
    for line in open(pair_file):
        tmp = line.rstrip().split()
        path_set.add(tmp[0])
        path_set.add(tmp[1])

    model, get_Conv_FeatureMap = load_model(layer_index=-5)
    for pic_path in path_set:
        fine_tune_feature = extract(pic_path, get_Conv_FeatureMap, (224, 224, 3))
        verif_path_feature_dic[pic_path] = fine_tune_feature
    msgpack_numpy.dump(verif_path_feature_dic, open(feature_pack_file, 'wb'))


def main_distance():
    all_data = []
    all_label = []
    all_pic_path_list = []
    count = 0
    verif_path_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    for line in open(pair_file):
        if count % 100 == 0:
            print count
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            path1 = tmp[0]
            path2 = tmp[1]
            label = int(tmp[2])
            feature1 = verif_path_feature_dic.get(path1)
            feature2 = verif_path_feature_dic.get(path2)
            # pdb.set_trace()
            predicts = pw.cosine_similarity(feature1, feature2)
            # predicts = np.fabs(feature1-feature2)
            all_data.append(predicts)
            all_label.append(label)
            all_pic_path_list.append((path1, path2))

    data = np.asarray(all_data)
    # print data.shape
    # data = np.reshape(data, newshape=(data.shape[0], 1))
    data = np.reshape(data, newshape=(data.shape[0], data.shape[2]))
    label = np.asarray(all_label)
    print data.shape, label.shape
    msgpack_numpy.dump((data, label, all_pic_path_list), open('original_verif_fc7_finetune_fc8.p', 'wb'))


def feature_fusion():
    kf = KFold(n_folds=10)
    all_acc = []
    (data, label, all_pic_path_list) = msgpack_numpy.load(open('original_verif_fc7_finetune_fc8.p', 'rb'))
    error_file = 'error_pair.txt'
    f = open(error_file, 'w')
    all_pic_path_list = np.asarray(all_pic_path_list)

    for k, (train, valid) in enumerate(kf.split(data, label)):
        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]
        valid_path = all_pic_path_list[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label,  clf.predict(valid_data))
        roc_auc = roc_auc_score(valid_label,  clf.predict(valid_data))
        # for index in range(len(valid_data)):
        #     if clf.predict(valid_data[index:index+1]) != valid_label[index]:
        #         f.write(valid_path[index][0]+'\t'+valid_path[index][1]+'\n')

        # rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=15)
        # rf_clf.fit(train_data, train_label)
        # rf_predict_train_label_prob = rf_clf.predict_proba(train_data)
        # rf_predict_valid_label_prob = rf_clf.predict_proba(valid_data)
        #
        # gb_clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=500)
        # gb_clf.fit(train_data, train_label)
        # gb_predict_train_label_prob = gb_clf.predict_proba(train_data)
        # gb_predict_valid_label_prob = gb_clf.predict_proba(valid_data)
        # mf_clf = RandomForestClassifier()
        # mf_train_data = np.column_stack((rf_predict_train_label_prob, gb_predict_train_label_prob))
        # mf_valid_data = np.column_stack((rf_predict_valid_label_prob, gb_predict_valid_label_prob))
        # mf_clf.fit(mf_train_data, train_label)
        # acc = accuracy_score(valid_label, mf_clf.predict(mf_valid_data))
        # roc_auc = roc_auc_score(valid_label, mf_clf.predict(mf_valid_data))
        # acc = accuracy_score(valid_label, rf_clf.predict(valid_data))
        # roc_auc = roc_auc_score(valid_label, rf_clf.predict(valid_data))
        # acc = accuracy_score(valid_label, gb_clf.predict(valid_data))
        # roc_auc = roc_auc_score(valid_label, gb_clf.predict(valid_data))

        all_acc.append(acc)
        print acc, roc_auc

        # roc_auc = roc_auc_score(valid_label, clf.predict(valid_data))
        # print acc, roc_auc
        # cPickle.dump(clf, open('/data/liubo/face/vgg_face_dataset/model/lfw_verification_model', 'wb'))
    print 'mean :', np.mean(all_acc)
    f.close()


if __name__ == '__main__':
    feature_pack_file = '/data/liubo/face/picture_feature/original_verif_fc7_finetune_feature.p'
    pair_file = '/data/liubo/face/originalimages/originalimages_verif_pair.txt'
    # pair_file = '/data/liubo/face/self_all_pair.txt'
    # pair_file = '/data/liubo/face/self_all_pair_0.5_0.5.txt'
    pair_file = '/data/liubo/face/self_all_pair_0.5_0.37.txt'
    extract_verif_feature()
    main_distance()
    feature_fusion()


# pw.cosine_similarity(feature1, feature2)  +   LinearSVC()  +  layer_index=-5
# 0.854838709677 0.852083333333
# 0.91935483871 0.919791666667
# 0.935483870968 0.935416666667
# 0.951612903226 0.954545454545
# 0.903225806452 0.898641588297
# 0.91935483871 0.91875
# 0.819672131148 0.816810344828
# 0.83606557377 0.837284482759
# 0.852459016393 0.854301075269
# 0.901639344262 0.90625
# mean : 0.889370703332

# np.fabs(feature1-feature2)  +  RandomForestClassifier(n_estimators=500, n_jobs=15)  +  layer_index=-5
# 0.854838709677 0.852083333333
# 0.870967741935 0.875
# 0.951612903226 0.95
# 0.854838709677 0.849007314525
# 0.91935483871 0.920062695925
# 0.983870967742 0.984375
# 0.852459016393 0.856142241379
# 0.786885245902 0.775862068966
# 0.704918032787 0.708064516129
# 0.918032786885 0.921875
# mean : 0.869777895293

# np.fabs(feature1-feature2)  +  GradientBoostingClassifier(learning_rate=0.05, n_estimators=500)  +  layer_index=-5
# 0.838709677419 0.835416666667
# 0.838709677419 0.839583333333
# 0.951612903226 0.951041666667
# 0.903225806452 0.898641588297
# 0.903225806452 0.902821316614
# 0.935483870968 0.9375
# 0.803278688525 0.802801724138
# 0.803278688525 0.797952586207
# 0.852459016393 0.85376344086
# 0.934426229508 0.93588362069
# mean : 0.876441036489