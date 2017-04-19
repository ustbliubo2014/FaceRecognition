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
from ..util.DeepId import euclidean_distance, eucl_dist_output_shape, contrastive_loss, create_pair_data, compute_accuracy


def conv_layer(pre_layer, nb_filter):
    current_layer = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pre_layer)
    current_layer = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, activation='relu', border_mode='same')(current_layer)
    current_layer = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, activation='relu', border_mode='same')(current_layer)
    current_layer = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, activation='relu', border_mode='same')(current_layer)
    current_layer = MaxPooling2D(pool_size=(2, 2))(current_layer)
    current_layer_fc = Flatten()(current_layer)
    return current_layer, current_layer_fc


def build_deepid3_model(input_shape, nb_classes):
    print('building deepid3 model')
    # 每层做成一个share model, 每层都计算distance, 各个loss的权重参考DeepId2里的设置
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    layer1_a, layer1_fc_a = conv_layer(input_a, nb_filter=32)
    layer1_b, layer1_fc_b = conv_layer(input_b, nb_filter=32)
    distance_fc1 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([layer1_fc_a, layer1_fc_b])

    layer2_a, layer2_fc_a = conv_layer(layer1_a, nb_filter=64)
    layer2_b, layer2_fc_b = conv_layer(layer1_b, nb_filter=64)
    distance_fc2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([layer2_fc_a, layer2_fc_b])

    layer3_a, layer3_fc_a = conv_layer(layer2_a, nb_filter=128)
    layer3_b, layer3_fc_b = conv_layer(layer2_b, nb_filter=128)
    distance_fc3 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([layer3_fc_a, layer3_fc_b])

    layer4_a, layer4_fc_a = conv_layer(layer3_a, nb_filter=256)
    layer4_b, layer4_fc_b = conv_layer(layer3_b, nb_filter=256)
    distance_fc4 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([layer4_fc_a, layer4_fc_b])

    pred_a = Dense(nb_classes, activation='softmax')(layer4_fc_a)
    pred_b = Dense(nb_classes, activation='softmax')(layer4_fc_b)

    model = Model(input=[input_a, input_b], output=[distance_fc1, distance_fc2, distance_fc3, distance_fc4, pred_a, pred_b])
    opt = Adagrad(lr=0.01)
    model.compile(optimizer=opt,
                  loss=[contrastive_loss, contrastive_loss, contrastive_loss, contrastive_loss,
                        'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[0.05, 0.05, 0.05, 0.05, 0.5, 0.5])
    return model


def train_valid_deepid3(X_train, y_train, X_test, y_test, nb_classes, model_file, weight_file):

    input_shape = X_train.shape[1:]
    tr_pairs, tr_y, X_train_first, y_train_first, X_train_second, y_train_second, te_pairs, te_y, X_test_first, \
                    y_test_first, X_test_second, y_test_second = create_pair_data(X_train, y_train, X_test, y_test, nb_classes)
    y_train_first = np_utils.to_categorical(y_train_first, nb_classes)
    y_train_second = np_utils.to_categorical(y_train_second, nb_classes)
    y_test_first = np_utils.to_categorical(y_test_first, nb_classes)
    y_test_second = np_utils.to_categorical(y_test_second, nb_classes)

    model = build_deepid3_model(input_shape, nb_classes)
    open(model_file,'w').write(model.to_json())
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
    checkpointer = ModelCheckpoint(weight_file, monitor='val_dense_2_loss', save_best_only=True)
    
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], [tr_y, tr_y, tr_y, tr_y, y_train_first, y_train_second],
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], [te_y, te_y, te_y, te_y, y_test_first, y_test_second]),
            batch_size=40, nb_epoch=10,
            callbacks=[checkpointer]
    )

    start = time()
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    # pdb.set_trace()
    te_acc = compute_accuracy(pred[0], te_y)
    first_pred=np.argmax(pred[-2], axis=1)
    first_true=np.argmax(y_test_first, axis=1)
    first_classify_acc = accuracy_score(first_true, first_pred)

    second_pred=np.argmax(pred[-1], axis=1)
    second_true=np.argmax(y_test_second, axis=1)
    second_classify_acc = accuracy_score(second_true, second_pred)

    end = time()
    print 'all predict time',(end -start)
    print('contrastive Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('classify Accuracy on second test set: %0.2f%%' % (100 * first_classify_acc))
    print('classify Accuracy on second test set: %0.2f%%' % (100 * second_classify_acc))



def extract_feature(pic_data, model_file, weight_file):
    # deepid2的特征提取,需要修改
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

