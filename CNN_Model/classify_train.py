# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: classify_train.py
@time: 2016/8/8 16:00
@contact: ustb_liubo@qq.com
@annotation: classify_train
"""
import sys
import logging
from logging.config import fileConfig
import os
import msgpack_numpy
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.callbacks import ModelCheckpoint

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

def build_classify(feature_dim, class_num):
    input_layer = Input(shape=(feature_dim,), name='input')
    hidden_layer = Dense(2048, activation='relu', name='hidden_layer1')(input_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)
    hidden_layer = Dense(1024, activation='relu', name='hidden_layer2')(hidden_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)
    classify_layer = Dense(class_num, activation='softmax', name='classify_layer')(hidden_layer)
    # classify_layer = Dense(class_num, activation='softmax', name='classify_layer')(input_layer)
    model = Model(input=[input_layer], output=[classify_layer])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    person_feature_list_dic = msgpack_numpy.load(open('/data/pictures_annotate_feature/person_feature_list_dic.p', 'rb'))
    from util import load_label_data
    train_data, valid_data, train_label, valid_label = load_label_data(person_feature_list_dic)
    # data, label = msgpack_numpy.load(open('/data/pictures_annotate_feature/more_person_with_self_data_label.p', 'rb'))
    # data = np.reshape(data, newshape=(data.shape[0], data.shape[2]))
    class_num = len(set(train_label))
    model = build_classify(feature_dim=train_data.shape[1], class_num=class_num)
    train_label = np_utils.to_categorical(train_label, class_num)
    valid_label = np_utils.to_categorical(valid_label, class_num)
    model_file = '/data/liubo/face/vgg_face_dataset/model/annotate_mlp_classify_%d.model'%class_num
    weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate_mlp_classify_%d.weight'%class_num


    checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, monitor='val_acc', save_best_only=True)
    model.fit(train_data, train_label, batch_size=128, nb_epoch=30, verbose=0, shuffle=True,
              validation_data=(valid_data, valid_label), callbacks=[checkpointer])
    score = model.evaluate(valid_data, valid_label, verbose=0)
    print score
    model.save_weights(weight_file, overwrite=True)
    open(model_file,'w').write(model.to_json())
