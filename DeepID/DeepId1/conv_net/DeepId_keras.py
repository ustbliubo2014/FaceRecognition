#-*- coding:utf-8 -*-
__author__ = 'liubo-it'


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Graph, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from load_data import load_numpy_data
import os
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import msgpack_numpy
import pdb


batch_size = 32
nb_epoch = 50
nb_filters = 32
nb_pool = 2
train_image_file = '/data/liubo/face/youtube/train_valid_data/train_train_images.p'
test_image_file = '/data/liubo/face/youtube/train_valid_data/train_test_images.p'
# train_image_file = '/home/data/dataset/images/youtube/patch_all_data/59/train_numpy_images.p'
# test_image_file = '/home/data/dataset/images/youtube/patch_all_data/59/test_numpy_images.p'
model_file = '/data/liubo/face/youtube/train_valid_data/train_images.model'
# model_file = '/home/data/dataset/images/youtube/patch_all_data/59/train_images.model'
weight_file = '/data/liubo/face/youtube/train_valid_data/train_images.weight'
# weight_file = '/home/data/dataset/images/youtube/patch_all_data/59/train_images.weight'
# train_threshold_sim_file = '/home/data/dataset/images/youtube/train_valid_data/sim_threshold_train_images.p'
# test_threshold_sim_file = '/home/data/dataset/images/youtube/train_valid_data/sim_threshold_test_images.p'


def load_face_data(train_image_file, test_image_file, load_image=True):
    if load_image:
        X_train, Y_train, X_test, Y_test = load_numpy_data(train_image_file, test_image_file)
    else:
        X_train, Y_train, X_test, Y_test = load_numpy_data(train_image_file, test_image_file)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = len(set(list(Y_train)) | set(list(Y_test)))
    pdb.set_trace()
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
    # (7260, 3, 55, 47) (7260, 363) (1815, 3, 55, 47) (1815, 363)
    return X_train, Y_train, X_test, Y_test

def load_mnist_data():
    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    # (60000, 1, 28, 28)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_train,Y_train,X_test,Y_test


def build_Graph_model(channel_num, img_rows, img_cols, nb_classes):
    print('building graph model')
    graph = Graph()
    graph.add_input(name='input', input_shape=(channel_num, img_rows, img_cols))
    graph.add_node(Convolution2D(nb_filters, 3, 3, border_mode='same'),name='Conv1', input='input')
    graph.add_node(MaxPooling2D(pool_size=(nb_pool, nb_pool)), name='pooling1', input='Conv1')
    graph.add_node(Convolution2D(nb_filters, 3, 3, border_mode='same'), name='Conv2', input='pooling1')
    graph.add_node(MaxPooling2D(pool_size=(nb_pool, nb_pool)), name='pooling2', input='Conv2')
    graph.add_node(Convolution2D(nb_filters, 3, 3, border_mode='same'), name='Conv3', input='pooling2')
    graph.add_node(MaxPooling2D(pool_size=(nb_pool, nb_pool)), name='pooling3', input='Conv3')
    graph.add_node(Convolution2D(nb_filters, 3, 3, border_mode='same'), name='Conv4', input='pooling3')

    # graph.add_node(Flatten(), name='flatten', inputs=['pooling3', 'Conv4'], merge_mode='concat')
    graph.add_node(Flatten(), name='flatten1', input='Conv4')
    graph.add_node(Flatten(), name='flatten2', input='pooling3')
    graph.add_node(Dense(output_dim=160, activation='relu'), name='hidden', inputs=['flatten1','flatten2'])
    graph.add_node(Dropout(0.2), name='dropout', input='hidden')
    graph.add_node(Dense(activation='softmax', output_dim=nb_classes), name='softmax',input='dropout')
    graph.add_output(name='output', input='softmax')
    opt = SGD(lr=0.01, momentum=0.9)
    graph.compile(opt, {'output':'categorical_crossentropy'})

    return graph


def main_graph(train_image_file, test_image_file, model_file, weight_file):
    X_train, Y_train, X_test, Y_test = load_face_data(train_image_file, test_image_file)
    model = build_Graph_model(X_train.shape[1], X_train.shape[2], X_train.shape[3], Y_train.shape[1])
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
    open(model_file,'w').write(model.to_json())
    model.fit({'input':X_train, 'output':Y_train}, nb_epoch=nb_epoch, verbose=1,
              validation_data={'input':X_test, 'output':Y_test})
    predict = model.predict({'input':X_test, 'output':Y_test}, verbose=0).get('output')
    print('predict.shape',predict.shape)
    score = model.evaluate({'input':X_test, 'output':Y_test}, verbose=1)
    print('Test score:', score)
    Y_predict = np.argmax(predict, axis=1)
    Y_test_1 = np.argmax(Y_test, axis=1)
    result = [1 if Y_predict[index] == Y_test_1[index] else 0 for index in range(Y_predict.shape[0])]
    print(np.sum(result)*1.0/Y_predict.shape[0])
    model.save_weights(weight_file,overwrite=True)


def extract_feature(train_image_file, test_image_file, model_file, weight_file, extract_feature_file, load_image=True):
    model = model_from_json(open(model_file).read())
    model.load_weights(weight_file)
    get_Conv_FeatureMap = theano.function([model.inputs['input'].get_input(train=False)],
                                          model.nodes.get('hidden').get_output(train=False))
    X_train, Y_train, X_test, Y_test = load_face_data(train_image_file, test_image_file, load_image=load_image)
    X_train_feature = get_Conv_FeatureMap(X_train)
    X_test_feature = get_Conv_FeatureMap(X_test)
    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
    # msgpack_numpy.dump((X_train_feature, Y_train, X_test_feature, Y_test), open(extract_feature_file,'wb'))


def extract_lfw_feature(model_file, weight_file, lfw_image_file, extract_feature_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(weight_file)
    get_Conv_FeatureMap = theano.function([model.inputs['input'].get_input(train=False)],
                                          model.nodes.get('hidden').get_output(train=False))
    X_train, Y_train, X_test, Y_test = msgpack_numpy.load(open(lfw_image_file,'rb'))
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    print 'lfw',X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
    X_train_feature = get_Conv_FeatureMap(X_train)
    X_test_feature = get_Conv_FeatureMap(X_test)
    # Y_train = np.argmax(Y_train, axis=1)
    # Y_test = np.argmax(Y_test, axis=1)
    msgpack_numpy.dump((X_train_feature, Y_train, X_test_feature, Y_test), open(extract_feature_file,'wb'))


def test_feature(X_train_feature, Y_train, X_test_feature, Y_test):
    # X_train_feature, Y_train, X_test_feature, Y_test = msgpack_numpy.load(open('sim_threshold_feature.p', 'rb'))
    clf = RandomForestClassifier(n_estimators=75, n_jobs=10)
    clf.fit(X_train_feature, Y_train)
    Y_predict = clf.predict(X_test_feature)
    print 'accuracy_score', accuracy_score(Y_test, Y_predict)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # linear的效果最好
    for kernel in kernels:
        all_C = [10, 1, 0.1, 0.01]
        for C in all_C:
            clf = SVC(C=C, kernel=kernel)
            clf.fit(X_train_feature, Y_train)
            Y_predict = clf.predict(X_test_feature)
            print 'C', C, 'kernel', kernel, 'accuracy_score', accuracy_score(Y_test, Y_predict)

if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = load_face_data(train_image_file, test_image_file)

    # # if len(sys.argv) != 6:
    # #     print 'Usage: python %s train_image_file test_image_file model_file weight_file' % (sys.argv[0])
    # # train_image_file = sys.argv[1]
    # # test_image_file = sys.argv[2]
    # # model_file = sys.argv[3]
    # # weight_file = sys.argv[4]
    # # extract_feature_file = sys.argv[5]
    # # main_graph(train_image_file, test_image_file, model_file, weight_file)
    # # extract_feature(train_image_file, test_image_file, model_file, weight_file, extract_feature_file='')
    # # #
    # # model_file = '/home/data/dataset/images/youtube/train_valid_data/train_images.model'
    # # weight_file = '/home/data/dataset/images/youtube/train_valid_data/train_images.weight'
    # lfw_image_file = '/home/data/dataset/images/lfw_data/train_valid_data.p'
    # # extract_feature_file = '/home/data/dataset/images/lfw_data/train_valid_feature.p'
    # # # extract_lfw_feature(model_file, weight_file, lfw_image_file, extract_feature_file)
    # # (X_train_feature, Y_train, X_test_feature, Y_test) = msgpack_numpy.load(open(extract_feature_file,'rb'))
    # # test_feature(X_train_feature, Y_train, X_test_feature, Y_test)
    #
    # X_train, Y_train, X_test, Y_test = msgpack_numpy.load(open(lfw_image_file,'rb'))
    # nb_classes = len(set(list(Y_train)) | set(list(Y_test)))
    # Y_train = np_utils.to_categorical(Y_train, nb_classes)
    # Y_test = np_utils.to_categorical(Y_test, nb_classes)
    # X_train = np.transpose(X_train, (0, 3, 1, 2))
    # X_test = np.transpose(X_test, (0, 3, 1, 2))
    # print 'lfw', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
    # model = build_Graph_model(X_train.shape[1], X_train.shape[2], X_train.shape[3], Y_train.shape[1])
    #
    # model.fit({'input':X_train, 'output':Y_train}, nb_epoch=3, verbose=1,
    #           validation_data={'input':X_test, 'output':Y_test})
    # predict = model.predict({'input':X_train, 'output':Y_train}, verbose=0).get('output')
    # print('predict.shape',predict.shape)
    # score = model.evaluate({'input':X_train, 'output':Y_train}, verbose=1)
    # print('Test score:', score)
    # Y_predict = np.argmax(predict, axis=1)
    # Y_test_1 = np.argmax(Y_train, axis=1)
    # count_dic = {}
    # for y in Y_test_1:
    #     count_dic[y] = count_dic.get(y,0) + 1
    # print count_dic
    # print set(list(Y_predict))
    # result = [1 if Y_predict[index] == Y_test_1[index] else 0 for index in range(Y_predict.shape[0])]
    # print(np.sum(result)*1.0/Y_predict.shape[0])
