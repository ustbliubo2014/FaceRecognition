# -*- coding:utf-8 -*-
__author__ = 'liubo-it'


import numpy as np
import pdb
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Input, Lambda, merge, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import os
from time import time
import theano
from ..util.DeepId import create_deepId_network
from ..util.util import get_top5_acc
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
from keras import backend as K


def build_deepid_model(create_base_network, input_shape, nb_classes):
    print('building deepid model')
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    processed_a = base_network(input_a)
    pred_a = Dense(nb_classes, activation='softmax')(processed_a)
    model = Model(input=[input_a], output=[pred_a])
    # opt = SGD(lr=0.01)
    opt = RMSprop()
    # opt = Adagrad()
    # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
    model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    return model


def train_valid_deepid(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file,
                       error_train_sample_file, error_valid_sample_file, path_list_test=None):
    '''
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param nb_classes:
    :param model_file:
    :param weight_file:
    :param error_train_sample_file:
    :param error_valid_sample_file:
    :param path_list_test: 测试数据对应的文件名(用于分析模型)
    :return:
    '''

    input_shape = X_train.shape[1:]
    # pdb.set_trace()
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if not os.path.exists(model_file):
        model = build_deepid_model(create_deepId_network, input_shape, nb_classes)
        open(model_file,'w').write(model.to_json())
    else:
        print 'load model'
        model = model_from_json(open(model_file, 'r').read())
        # opt = SGD()
        # opt = RMSprop()
        opt = Adam()
        model.compile(optimizer=opt, loss=['categorical_crossentropy'])
    if os.path.exists(weight_file):
        print 'load_weights'
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
    batch_size = 128
    Y_predict_batch = model.predict(X_test, batch_size=batch_size, verbose=1)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(Y_predict_batch, axis=1))
    test_acc = np.min([test_acc, 0.7])
    last_crps = test_acc

    print('last_crps :', last_crps)
    this_patience = 0
    patience = 10
    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print('Training...')
        progbar = generic_utils.Progbar(X_train.shape[0])
        sample_num = 0
        # shuffle 数据
        length = X_train.shape[0]
        shuffle_list = range(length)
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
        Y_train_preidct_batch = model.predict(X_train, batch_size=batch_size, verbose=1)
        train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(Y_train_preidct_batch, axis=1))
        print ('train_acc :', train_acc,  'test acc', test_acc)
        if last_crps < test_acc:
            this_patience = 0
            model.save_weights(weight_file, overwrite=True)
            print ('save_model')
            last_crps = test_acc
        else:
            if this_patience >= patience:
                break
            else:
                this_patience = 1

        start = time()
        test_pred = model.predict([X_test])
        test_true = np.argmax(y_test, axis=1)
        test_top1_acc = accuracy_score(test_true, np.argmax(test_pred, axis=1))
        test_pred = np.argmax(test_pred, axis=1)
        train_pred = np.argmax(train_pred, axis=1)
        f_error_valid_sample = open(error_valid_sample_file,'w')
        f_error_valid_sample.write('index''\t''true''\t''predict''\n')
        for index in range(len(test_true)):
            if test_pred[index] != test_true[index]:
                f_error_valid_sample.write('\t'.join(map(str,[index, test_true[index], test_pred[index]]))+'\n')


def train_deepid_batch(X_train, y_train, nb_classes, model, weight_file):
    '''
        大规模数据集,多次读取数据
    '''
    # input_shape = X_train.shape[1:]
    # y_train = np_utils.to_categorical(y_train, nb_classes)
    # y_test = np_utils.to_categorical(y_test, nb_classes)
    #
    # model = build_deepid_model(create_deepId_network, input_shape, nb_classes)
    # open(model_file,'w').write(model.to_json())
    # if os.path.exists(weight_file):
    #     print 'load_weights'
    #     model.load_weights(weight_file)
    #
    # pdb.set_trace()
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
    nb_epoch = 1
    batch_size = 128
    for e in range(nb_epoch):
        print('Training...')
        progbar = generic_utils.Progbar(X_train.shape[0])
        sample_num = 0
        for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('train loss', loss)])
            sample_num += X_batch.shape[0]
            if sample_num >= X_train.shape[0]:
                break


def valid_deepid_batch(X_test, y_test, model, weight_file):
    '''
        大规模数据集,多次读取数据
    '''
    Y_predict_batch = model.predict(X_test, batch_size=128, verbose=1)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(Y_predict_batch, axis=1))
    return test_acc


def create_extract_func(model_file, weight_file):
    model = model_from_json(open(model_file,'r').read())
    model.load_weights(weight_file)
    # pdb.set_trace()
    start = time()
    # get_Conv_FeatureMap = theano.function([model.layers[1].layers[0].get_input_at(False)],
    #                                   model.layers[1].layers[-1].get_output_at(False))
    get_Conv_FeatureMap = K.function([model.layers[1].layers[0].get_input_at(False), K.learning_phase()],
                                     [model.layers[1].layers[-1].get_output_at(False)])
    end = time()
    print 'create func time :', (end-start)
    return get_Conv_FeatureMap

def extract_feature(pic_data, model_file, weight_file,feature_dim=512):
    print 'pic_data.shape :', pic_data.shape, 'feature_dim :', feature_dim

    get_Conv_FeatureMap = create_extract_func(model_file, weight_file)
    start = time()
    # pic_data可以会很大,直接运行会内存报错,修改成分批处理
    batch_size = 1
    batch_num = pic_data.shape[0] / batch_size
    pic_data_feature = np.zeros(shape=(pic_data.shape[0], feature_dim))
    for num in range(batch_num):
        batch_data = pic_data[num*batch_size:(num+1)*batch_size, :, :, :]
        pic_data_feature[num*batch_size:(num+1)*batch_size, :] = get_Conv_FeatureMap([batch_data[:,:],0])[0]
    end = time()
    print 'pic_data_feature.shape : ', pic_data_feature.shape, ' extract feature time : ', (end -start)
    return pic_data_feature


def main():
    # model = build_deepid_model(create_deepId_network, (3,50,50), 2622)
    import msgpack_numpy
    from keras import backend as K
    x,y = msgpack_numpy.load(open('/home/liubo-it/FaceRecognization/DeepId_data.p','rb'))
    pdb.set_trace()
    model = build_deepid_model(create_deepId_network, (3,50,50), 2622)
    pdb.set_trace()
    get_Conv_FeatureMap = K.function([model.layers[1].layers[0].get_input_at(False),
                                      K.learning_phase()], [model.layers[1].layers[-1].get_output_at(False)])

    # output in test mode = 0
    layer_output = get_Conv_FeatureMap([x, 0])[0]

    # output in train mode = 1
    layer_output = get_Conv_FeatureMap([x, 1])[0]