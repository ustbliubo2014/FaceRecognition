# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: fine_tune.py
@time: 2016/8/29 19:24
@contact: ustb_liubo@qq.com
@annotation: fine_tune
"""
import sys
import logging
from logging.config import fileConfig
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
import os
from keras.optimizers import Adagrad, Adam, SGD
from keras import backend as K
from keras.layers.convolutional import ZeroPadding2D
sys.path.append('/home/liubo-it/FaceRecognization')
from extract_annotate_feature import load_train_data
from sklearn.cross_validation import train_test_split
import pdb
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


# 把数据压缩到一个文件(利用extract里的函数,注意减去初值)
# 以merge_model的参数为初始值,重新训练模型(看是否会过拟合)
def train_valid_model(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file):
    input_shape = X_train.shape[1:]
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print 'load model'
    model = model_from_json(open(model_file, 'r').read())
    opt = SGD(1e-4)
    # opt = RMSprop()
    # opt = Adam()
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    model.load_weights(weight_file)

    datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
    datagen.fit(X_train, augment=False)
    nb_epoch = 500
    batch_size = 64
    test_loss = []
    batch_num = X_test.shape[0] / batch_size
    for index in range(batch_num):
        try:
            tmp = model.test_on_batch(X_test[batch_size*index:(index+1)*batch_size], y_test[batch_size*index:(index+1)*batch_size])
            test_loss.append(tmp)
        except:
            traceback.print_exc()
            continue
    last_loss = np.mean(test_loss)

    length = X_train.shape[0]
    shuffle_list = range(length)
    print('last_loss :', last_loss)
    this_patience = 0
    patience = 10

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print('Training...')
        progbar = generic_utils.Progbar(length)
        sample_num = 0
        # 每次手动shuffle

        np.random.shuffle(shuffle_list)
        X_train = X_train[shuffle_list]
        y_train = y_train[shuffle_list]
        for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('train loss', loss)])
            sample_num += X_batch.shape[0]
            if sample_num >= X_train.shape[0]:
                break

        print('Testing...')
        Y_predict_batch = model.predict(X_test, batch_size=batch_size, verbose=1)
        test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(Y_predict_batch, axis=1))
        # Y_train_preidct_batch = model.predict(X_train, batch_size=batch_size, verbose=1)
        # train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(Y_train_preidct_batch, axis=1))

        test_loss = []
        for index in range(batch_num):
            tmp = model.test_on_batch(X_test[batch_size*index:(index+1)*batch_size], y_test[batch_size*index:(index+1)*batch_size])
            test_loss.append(tmp)
        test_loss = np.mean(test_loss)
        print ('test acc :', test_acc, 'test_loss :', test_loss)
        # loss越低越好, acc越高越好
        if last_loss > test_loss:
            this_patience = 0
            model.save_weights(weight_file, overwrite=True)
            print ('save_model')
            last_loss = test_loss
        else:
            if this_patience >= patience:
                break
            else:
                this_patience = 1


if __name__ == '__main__':
    nb_class = 737
    data_folder = '/data/pictures_annotate'
    model_file = '/data/liubo/face/vgg_face_dataset/model/annotate_deep_face_%d.model'%nb_class
    weight_file = '/data/liubo/face/vgg_face_dataset/model/annotate_deep_face_%d.weight'%nb_class

    data, label = load_train_data(data_folder)
    train_data, valid_data, train_label, valid_label = train_test_split(data, label)
    print train_data.shape, valid_data.shape, train_label.shape, valid_label.shape
    train_valid_model(train_data, train_label, valid_data, valid_label, nb_class, model_file, weight_file)
