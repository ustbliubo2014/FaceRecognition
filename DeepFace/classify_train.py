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
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')

def build_classify(feature_dim, class_num):
    input_layer = Input(shape=(feature_dim,), name='input')
    hidden_layer = Dense(2048, activation='relu', name='hidden_layer1')(input_layer)
    # hidden_layer = Dropout(0.5)(hidden_layer)
    hidden_layer = Dense(1024, activation='relu', name='hidden_layer2')(hidden_layer)
    # hidden_layer = Dropout(0.5)(hidden_layer)
    classify_layer = Dense(class_num, activation='softmax', name='classify_layer')(hidden_layer)
    # classify_layer = Dense(class_num, activation='softmax', name='classify_layer')(input_layer)

    model = Model(input=[input_layer], output=[classify_layer])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    data, label = msgpack_numpy.load(open('/data/pictures_annotate_feature/annotate_data.p', 'rb'))
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_depth=None)
    clf.fit(X_train, Y_train)
    print accuracy_score(Y_test, clf.predict(X_test))

    class_num = len(set(label))
    model = build_classify(feature_dim=data.shape[1], class_num=class_num)
    label = np_utils.to_categorical(label, class_num)


    model_file = '/data/liubo/face/vgg_face_dataset/model/annotate_mlp_classify_%d.model'%class_num
    weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate_mlp_classify_%d.weight'%class_num

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1)
    checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, monitor='val_acc', save_best_only=True)

    model.fit(X_train, Y_train, batch_size=32, nb_epoch=100, verbose=1, shuffle=True, validation_data=(X_test, Y_test),
              callbacks=[checkpointer])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print score
    model.save_weights(weight_file, overwrite=True)
    open(model_file,'w').write(model.to_json())
