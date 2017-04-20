# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: lfw_keras_vgg_valid.py
@time: 2016/8/8 12:39
@contact: ustb_liubo@qq.com
@annotation: lfw_keras_vgg_valid
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
from sklearn.model_selection import KFold

avg = np.array([129.1863, 104.7624, 93.5940])
lfw_folder = '/data/liubo/face/lfw_face'
# lfw_folder = '/data/liubo/face/lfw_dlib_face'
pair_file = '/data/liubo/face/lfw_pair.txt'
feature_pack_file = '/data/liubo/face/lfw_vgg_feature.p'
# error_pair_file = '/data/liubo/face/error_pair.txt'
error_pair_file = '/data/liubo/face/error_dlib_pair.txt'
pic_shape = (224, 224, 3)


def read_one_rgb_pic(pic_path, pic_shape=pic_shape):
    img = imresize(imread(pic_path), pic_shape)
    img = img[:, :, ::-1]*1.0
    img = img - avg
    img = img.transpose((2, 0, 1))
    img = img[None, :]
    return img


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
        return model


model = load_model()
# ____________________________________________________________________________________________________
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
# 24 prob (Dense)                       (None, 2622)        10742334    dropout_2[0][0]
# ====================================================================================================
# Total params: 145002878
get_Conv_FeatureMap = K.function([model.layers[0].get_input_at(False), K.learning_phase()],
                                 [model.layers[22].get_output_at(False)])


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
                continue
        lfw_feature_dic[person] = this_person_feature_dic
    msgpack_numpy.dump(lfw_feature_dic, open(feature_pack_file, 'wb'))


def main_distance():
    # lfw_feature_dic = msgpack_numpy.load(open(feature_pack_file, 'rb'))
    lfw_feature_dic = msgpack_numpy.load(open('LightenedCNN.p', 'rb'))
    data = []
    label = []
    pic_path_list = []
    for line in open(pair_file):
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            person = tmp[0] #取该人的两个特征向量
            index1 = int(tmp[1])
            index2= int(tmp[2])
            this_person_feature_dic = lfw_feature_dic.get(person, {})
            if index1 in this_person_feature_dic and index2 in this_person_feature_dic:
                feature1, path1 = this_person_feature_dic[index1]
                feature2, path2 = this_person_feature_dic[index2]
                predicts = pw.cosine_similarity(feature1, feature2)
                # predicts = np.fabs(feature1 - feature2)
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
                # predicts = np.fabs(feature1 - feature2)
                label.append(1)
                data.append(predicts)
                pic_path_list.append('\t'.join([path1, path2]))

    # pdb.set_trace()
    data = np.asarray(data)
    # data = np.reshape(data, newshape=(data.shape[0], data.shape[-1]))
    data = np.reshape(data, newshape=(data.shape[0], 1))
    label = np.asarray(label)
    print data.shape, label.shape

    # f = open(error_pair_file, 'w')
    pic_path_list = np.asarray(pic_path_list)

    kf = KFold(n_folds=10)
    all_acc = []
    for k, (train, valid) in enumerate(kf.split(data, label, pic_path_list)):
    # for k in range(10):
    #     train_data, valid_data, train_label, valid_label, train_path_list, valid_path_list = \
    #         train_test_split(data, label, pic_path_list, test_size=0.1)
        train_data = data[train]
        valid_data = data[valid]
        train_label = label[train]
        valid_label = label[valid]
        train_path_list = pic_path_list[train]
        valid_path_list = pic_path_list[valid]

        clf = LinearSVC()
        clf.fit(train_data, train_label)
        # for index in range(len(valid_label)):
        #     if clf.predict(valid_data[index]) != valid_label[index]:
        #         f.write(valid_path_list[index]+'\n')

        acc = accuracy_score(valid_label, clf.predict(valid_data))
        roc_auc = roc_auc_score(valid_label, clf.predict(valid_data))

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
        # roc_auc = roc_auc_score(valid_label, mf_clf.predict(mf_valid_data))

        all_acc.append(acc)
        print acc, roc_auc

        # roc_auc = roc_auc_score(valid_label, clf.predict(valid_data))
        # print acc, roc_auc
        # cPickle.dump(clf, open('/data/liubo/face/vgg_face_dataset/model/lfw_verification_model', 'wb'))
    print np.mean(all_acc)


def verfi_two_pic(pic_path1, pic_path2):
    feature1 = extract(pic_path1)
    feature2 = extract(pic_path2)
    clf = cPickle.load(open('/data/liubo/face/vgg_face_dataset/model/verification_model', 'rb'))
    predicts = pw.cosine_similarity(feature1, feature2)
    result = clf.predict(predicts)
    print result


if __name__ == '__main__':
    # verfi_two_pic(sys.argv[1], sys.argv[2])
    #
    # extract_lfw_feature()
    main_distance()
