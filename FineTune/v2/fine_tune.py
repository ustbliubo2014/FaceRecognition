# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: fine_tune.py
@time: 2016/8/26 16:19
@contact: ustb_liubo@qq.com
@annotation: fine_tune
"""
import sys
import logging
from logging.config import fileConfig
import os
from load_data import load_one_deep_path, person_path_dic_trans
from extract_feature import extract, read_one_rgb_pic
from util import load_model
import pdb
from time import time
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import msgpack
import numpy as np
import msgpack_numpy

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


pic_shape = (224, 224, 3)
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
# ____________________________________________________________________________________________________

def main():
    folder = '/data/hanlin'
    person_path_dic = load_one_deep_path(folder)
    sample_list, person_num = person_path_dic_trans(person_path_dic)
    model, get_Conv_FeatureMap = load_model(output_layer_index=18)
    data = []
    label = []
    start = time()
    for pic_path, person_index in sample_list:
        feature_vector = extract(pic_path, get_Conv_FeatureMap, pic_shape)[0]
        data.append(feature_vector)
        label.append(person_index)
    end = time()
    print (end - start)
    msgpack_numpy.dump((data, label), open('hanlin.p', 'wb'))


if __name__ == '__main__':
    main()
    # data, label = msgpack_numpy.load(open('hanlin.p', 'rb'))
    # clf = GradientBoostingClassifier()
    # clf = RandomForestClassifier(n_estimators=1500, n_jobs=-1)
    # data = np.asarray(data)
    # label = np.asarray(label)
    # train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.2)
    # print train_data.shape, valid_data.shape
    # clf.fit(train_data, train_label)
    # print accuracy_score(valid_label, clf.predict(valid_data))
