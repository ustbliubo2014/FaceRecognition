# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: annotate_vgg_self_valid.py
@time: 2016/8/11 14:42
@contact: ustb_liubo@qq.com
@annotation: annotate_vgg_self_valid
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
from time import time
import xgboost as xgb
import msgpack
import cPickle

avg = np.array([129.1863, 104.7624, 93.5940])
# 测试采集数据
pair_file = '/data/liubo/face/self_all_pair.txt'
feature_pack_file = '/data/liubo/face/annotate_self_feature.p'
error_pair_file = '/data/liubo/face/self_error_pair.txt'
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
        print 'load weights'
        model.load_weights(weight_file)
        return model


model = load_model()
# Layer (type)                       Output Shape        Param #     Connected to
# ====================================================================================================
# 0 input (InputLayer)                 (None, 3, 224, 224) 0
# ____________________________________________________________________________________________________
# 1 conv1_1 (Convolution2D)            (None, 64, 224, 224)1792        input[0][0]
# ____________________________________________________________________________________________________
# 2 conv1_2 (Convolution2D)            (None, 64, 224, 224)36928       conv1_1[0][0]
# ____________________________________________________________________________________________________
# 3 pool1 (MaxPooling2D)               (None, 64, 112, 112)0           conv1_2[0][0]
# ____________________________________________________________________________________________________
# 4 conv2_1 (Convolution2D)            (None, 128, 112, 11273856       pool1[0][0]
# ____________________________________________________________________________________________________
# 5 conv2_2 (Convolution2D)            (None, 128, 112, 112147584      conv2_1[0][0]
# ____________________________________________________________________________________________________
# 6 pool2 (MaxPooling2D)               (None, 128, 56, 56) 0           conv2_2[0][0]
# ____________________________________________________________________________________________________
# 7 conv3_1 (Convolution2D)            (None, 256, 56, 56) 295168      pool2[0][0]
# ____________________________________________________________________________________________________
# 8 conv3_2 (Convolution2D)            (None, 256, 56, 56) 590080      conv3_1[0][0]
# ____________________________________________________________________________________________________
# 9 conv3_3 (Convolution2D)            (None, 256, 56, 56) 590080      conv3_2[0][0]
# ____________________________________________________________________________________________________
# 10 pool3 (MaxPooling2D)               (None, 256, 28, 28) 0           conv3_3[0][0]
# ____________________________________________________________________________________________________
# 11 conv4_1 (Convolution2D)            (None, 512, 28, 28) 1180160     pool3[0][0]
# ____________________________________________________________________________________________________
# 12 conv4_2 (Convolution2D)            (None, 512, 28, 28) 2359808     conv4_1[0][0]
# ____________________________________________________________________________________________________
# 13 conv4_3 (Convolution2D)            (None, 512, 28, 28) 2359808     conv4_2[0][0]
# ____________________________________________________________________________________________________
# 14 pool4 (MaxPooling2D)               (None, 512, 14, 14) 0           conv4_3[0][0]
# ____________________________________________________________________________________________________
# 15 conv5_1 (Convolution2D)            (None, 512, 14, 14) 2359808     pool4[0][0]
# ____________________________________________________________________________________________________
# 16 conv5_2 (Convolution2D)            (None, 512, 14, 14) 2359808     conv5_1[0][0]
# ____________________________________________________________________________________________________
# 17 conv5_3 (Convolution2D)            (None, 512, 14, 14) 2359808     conv5_2[0][0]
# ____________________________________________________________________________________________________
# 18 pool5 (MaxPooling2D)               (None, 512, 7, 7)   0           conv5_3[0][0]
# ____________________________________________________________________________________________________
# 19 flatten (Flatten)                  (None, 25088)       0           pool5[0][0]
# ____________________________________________________________________________________________________
# 20 fc6 (Dense)                        (None, 4096)        102764544   flatten[0][0]
# ____________________________________________________________________________________________________
# 21 dropout_1 (Dropout)                (None, 4096)        0           fc6[0][0]
# ____________________________________________________________________________________________________
# 22 fc7 (Dense)                        (None, 4096)        16781312    dropout_1[0][0]
# ____________________________________________________________________________________________________
# 23 dropout_2 (Dropout)                (None, 4096)        0           fc7[0][0]
# ____________________________________________________________________________________________________
# 24 fc8 (Dense)                        (None, 2048)        8390656     dropout_2[0][0]
# ____________________________________________________________________________________________________
# 25 dropout_3 (Dropout)                (None, 2048)        0           fc8[0][0]
# ____________________________________________________________________________________________________
# 26 fc9 (Dense)                        (None, 1024)        2098176     dropout_3[0][0]
# ____________________________________________________________________________________________________
# 27 dropout_4 (Dropout)                (None, 1024)        0           fc9[0][0]
# ____________________________________________________________________________________________________
# 28 prob (Dense)                       (None, 757)         775925      dropout_4[0][0]
# ====================================================================================================
# Total params: 145525301

get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                 [model.layers[20].get_output_at(False)])


def extract(pic_path):
    img = read_one_rgb_pic(pic_path, pic_shape)
    start = time()
    feature_vector = get_Conv_FeatureMap([img, 0])[0].copy()
    end = time()
    # print 'extract feature time :', (end - start)
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
    not_in_pair = cPickle.load(open('not_in_pair.p', 'r'))
    for line in open(pair_file):
        if count % 100 == 0:
            print count
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            path1 = tmp[0]
            path2 = tmp[1]
            pic_path1 = os.path.split(tmp[0])[1]
            pic_path2 = os.path.split(tmp[1])[1]

            # if (pic_path1, pic_path2) in not_in_pair:
            #     print 'not in pair'
            #     continue
            label = int(tmp[2])
            feature1 = extract(path1)
            feature2 = extract(path2)
            # print feature1.shape, feature2.shape
            # feature1 = np.reshape(feature1, newshape=(1, feature1.shape[0]))
            # feature2 = np.reshape(feature2, newshape=(1, feature2.shape[0]))
            predicts = pw.cosine_similarity(feature1, feature2)
            # predicts = (feature1 - feature2) * (feature1 - feature2) / (feature1 + feature2)
            # predicts = np.fabs(feature1 - feature2)
            # print feature1, feature2
            all_data.append(predicts)
            all_label.append(label)
            all_pic_path_list.append((pic_path1, pic_path2))
    msgpack_numpy.dump((all_data, all_label, all_pic_path_list), open(feature_pack_file, 'wb'))


def verfi_two_pic(pic_path1, pic_path2):
    feature1 = extract(pic_path1)
    feature2 = extract(pic_path2)
    clf = cPickle.load(open('/data/liubo/face/vgg_face_dataset/model/verification_model', 'rb'))
    predicts = pw.cosine_similarity(feature1, feature2)
    result = clf.predict(predicts)
    print result


if __name__ == '__main__':
    # f = open(pair_file, 'w')
    # for line in open('/data/liubo/face/self_vgg_all_pair_score.txt'):
    #     tmp = line.rstrip().split()
    #     first_person = tmp[0].split('146')[0].replace('lining', 'lining-s')
    #     second_person = tmp[1].split('146')[0].replace('lining', 'lining-s')
    #     pic_path0 = os.path.join('/data/liubo/face/tmp', first_person, tmp[0].rstrip())
    #     pic_path1 = os.path.join('/data/liubo/face/tmp', second_person, tmp[1].rstrip())
    #     f.write(pic_path0+'\t'+pic_path1+'\t'+tmp[2]+'\n')



    main_distance()

    (all_data, all_label, all_pic_path_list) = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    all_data = np.asarray(all_data)
    all_data = np.reshape(all_data, newshape=(all_data.shape[0], all_data.shape[2]))
    all_label = np.asarray(all_label)
    all_pic_path_list = np.asarray(all_pic_path_list)
    print all_data.shape, all_label.shape

    not_in_pair = cPickle.load(open('not_in_pair.p', 'r'))
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
        # print valid_path_list[0]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        acc = accuracy_score(valid_label, clf.predict(valid_data))

        # right = wrong = 0
        # for index in range(len(valid_data)):
        #     predict = clf.predict(valid_data[index:index+1])
        #     if predict != valid_label[index]:
        #         if (valid_path_list[index][0], valid_path_list[index][1]) in not_in_pair:
        #             continue
        #         else:
        #             wrong += 1
        #     else:
        #         right += 1
        #     acc = right * 1.0 / (right + wrong)
            # acc = accuracy_score(valid_label, clf.predict(valid_data))


        # dtrain = xgb.DMatrix(train_data, label=train_label)
        # dtest = xgb.DMatrix(valid_data, label=valid_label)
        # param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
        # param['nthread'] = 4
        # plst = param.items()
        # plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way plst += [('eval_metric', 'ams@0')]
        # num_round = 10
        # evallist  = [(dtest,'eval'), (dtrain,'train')]
        #
        # bst = xgb.train( plst, dtrain, num_round, evallist )
        # valid_predict = bst.predict(dtest, ntree_limit=bst.best_iteration)
        # train_predict = bst.predict(dtrain, ntree_limit=bst.best_iteration)
        # train_predict = np.reshape(train_predict, newshape=(train_predict.shape[0], 1))
        # valid_predict = np.reshape(valid_predict, newshape=(valid_predict.shape[0], 1))
        # r_clf = LinearSVC()
        # r_clf.fit(train_predict, train_label)
        # acc = accuracy_score(valid_label, r_clf.predict(valid_predict))

        # rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=15)
        # rf_clf.fit(train_data, train_label)
        # rf_predict_train_label_prob = rf_clf.predict_proba(train_data)
        # rf_predict_valid_label_prob = rf_clf.predict_proba(valid_data)
        # gb_clf = GradientBoostingClassifier(learning_rate=0.03, n_estimators=1000)
        # gb_clf.fit(train_data, train_label)
        # gb_predict_train_label_prob = gb_clf.predict_proba(train_data)
        # gb_predict_valid_label_prob = gb_clf.predict_proba(valid_data)
        # mf_clf = RandomForestClassifier()
        # mf_train_data = np.column_stack((rf_predict_train_label_prob, gb_predict_train_label_prob))
        # mf_valid_data = np.column_stack((rf_predict_valid_label_prob, gb_predict_valid_label_prob))
        # mf_clf.fit(mf_train_data, train_label)
        # acc = accuracy_score(valid_label, mf_clf.predict(mf_valid_data))
        # right = wrong = 0
        # for index in range(len(mf_valid_data)):
        #     predict = mf_clf.predict(mf_valid_data[index:index+1])
        #     if predict != valid_label[index]:
        #         if (valid_path_list[index][0], valid_path_list[index][1]) in not_in_pair:
        #             continue
        #         else:
        #             wrong += 1
        #     else:
        #         right += 1
        #     acc = right * 1.0 / (right + wrong)
        #     acc = accuracy_score(valid_label, clf.predict(valid_data))



        # for index in range(len(predict_label)):
        #     if valid_label[index] != predict_label[index]:
        #         f.write(valid_path_list[index][0].rstrip()+'\t'+valid_path_list[index][1].rstrip()+'\t'+
        #                 str(valid_label[index]).rstrip()+'\t'+str(predict_label[index]).rstrip()+'\n')

        all_acc.append(acc)
        print acc
    f.close()
    print 'mean_acc :', np.mean(all_acc)
