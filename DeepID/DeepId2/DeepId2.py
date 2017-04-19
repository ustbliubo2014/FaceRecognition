# -*- coding:utf-8 -*-
__author__ = 'liubo-it'


import numpy as np
import pdb
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Input, Lambda, merge, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import os
from time import time
import theano
from ..util.DeepId import euclidean_distance, eucl_dist_output_shape, contrastive_loss, create_pair_data, create_deepId_network
from ..util.util import get_top5_acc


def build_deepid2_model(create_base_network, input_shape, nb_classes):
    print('building deepid2 model')
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    pred_a = Dense(nb_classes, activation='softmax')(processed_a)
    pred_b = Dense(nb_classes, activation='softmax')(processed_b)
    # model = Model(input=[input_a, input_b], output=[distance, pred_a, pred_b])
    model = Model(input=[input_a, input_b], output=[distance])
    opt = Adam(lr=0.01)
    # 需要将softmax的loss加到contrastive_loss中,并指定每个loss的权重
    model.compile(optimizer=opt,
                  # loss=[contrastive_loss, 'categorical_crossentropy', 'categorical_crossentropy'],
                  # loss_weights=[0.05, 0.5, 0.5])
                  loss=[contrastive_loss])
    return model


def train_valid_deepid2(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file,
                        error_train_sample_file, error_valid_sample_file):

    input_shape = X_train.shape[1:]
    tr_pairs, tr_y, X_train_first, y_train_first, X_train_second, y_train_second, te_pairs, te_y, X_test_first, \
                    y_test_first, X_test_second, y_test_second = create_pair_data(X_train, y_train, X_test, y_test, nb_classes)
    y_train_first = np_utils.to_categorical(y_train_first, nb_classes)
    y_train_second = np_utils.to_categorical(y_train_second, nb_classes)
    y_test_first = np_utils.to_categorical(y_test_first, nb_classes)
    y_test_second = np_utils.to_categorical(y_test_second, nb_classes)

    model = build_deepid2_model(create_deepId_network, input_shape, nb_classes)

    open(model_file, 'w').write(model.to_json())
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
    checkpointer = ModelCheckpoint(weight_file, monitor='val_dense_2_loss', save_best_only=True)
    pdb.set_trace()
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], [tr_y, y_train_first, y_train_second],
                validation_data=([te_pairs[:, 0], te_pairs[:, 1]], [te_y, y_test_first, y_test_second]),
                batch_size=128, nb_epoch=10, callbacks=[checkpointer])

    start = time()
    test_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])[1]
    train_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])[1]

    train_true = np.argmax(y_train, axis=1)
    test_true = np.argmax(y_test, axis=1)
    train_top1_acc = accuracy_score(train_true, np.argmax(train_pred, axis=1))
    train_top5_acc = get_top5_acc(train_pred, train_true)
    test_top1_acc = accuracy_score(test_true, np.argmax(test_pred, axis=1))
    test_top5_acc = get_top5_acc(test_pred, test_true)
    test_pred = np.argmax(test_pred, axis=1)
    train_pred = np.argmax(train_pred, axis=1)
    f_error_valid_sample = open(error_valid_sample_file,'w')
    f_error_valid_sample.write('index'+'\t'+'true'+'\t'+'predict'+'\n')
    for index in range(len(test_true)):
        if test_pred[index] != test_true[index]:
            f_error_valid_sample.write('\t'.join(map(str,[index, test_true[index], test_pred[index]]))+'\n')
    f_error_train_sample = open(error_train_sample_file,'w')
    f_error_train_sample.write('index'+'\t'+'true'+'\t'+'predict'+'\n')
    for index in range(len(train_true)):
        if train_pred[index] != train_true[index]:
            f_error_train_sample.write('\t'.join(map(str,[index, train_true[index], train_pred[index]]))+'\n')
    end = time()
    print 'all predict time',(end -start)
    print 'train_top1_acc : ', train_top1_acc, ' train_top5_acc : ', train_top5_acc, ' test_top1_acc : ', \
        test_top1_acc, ' test_top5_acc : ', test_top5_acc

    # start = time()
    # pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    # te_acc = compute_accuracy(pred[0], te_y)
    #
    # first_pred=np.argmax(pred[1], axis=1)
    # first_true=np.argmax(y_test_first, axis=1)
    # first_classify_acc = accuracy_score(first_true, first_pred)
    #
    # second_pred=np.argmax(pred[2], axis=1)
    # second_true=np.argmax(y_test_second, axis=1)
    # second_classify_acc = accuracy_score(second_true, second_pred)
    #
    # end = time()
    # print 'all predict time',(end -start)
    # print('contrastive Accuracy on test set: %0.2f%%' % (100 * te_acc))
    # print('classify Accuracy on second test set: %0.2f%%' % (100 * first_classify_acc))
    # print('classify Accuracy on second test set: %0.2f%%' % (100 * second_classify_acc))


def extract_feature(pic_data, model_file, weight_file):
    model = model_from_json(open(model_file,'r').read())
    model.load_weights(weight_file)
    # pdb.set_trace()
    get_Conv_FeatureMap = theano.function([model.layers[2].layers[0].get_input_at(False)],
                                          model.layers[2].layers[-1].get_output_at(False))
    start = time()
    pic_data_feature = get_Conv_FeatureMap(pic_data)
    end = time()
    print 'pic_data.shape : ', pic_data.shape, ' extract feature time : ', (end -start)
    return pic_data_feature

