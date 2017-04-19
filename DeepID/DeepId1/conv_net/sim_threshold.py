#-*- coding:utf-8 -*-
__author__ = 'liubo-it'


import numpy as np
np.random.seed(1337)  # for reproducibility
import msgpack_numpy


def cal_vector_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1))*np.sqrt(np.dot(vec2,vec2)))


def cal_sim_threshold(X_train_feature, Y_train):
    all_sim = np.zeros((X_train_feature.shape[0], X_train_feature.shape[0]))
    for index_i in range(all_sim.shape[0]):
        for index_j in range(index_i+1, all_sim.shape[1]):
            all_sim[index_i][index_j] = \
                cal_vector_sim(X_train_feature[index_i], X_train_feature[index_j])
    right = wrong = 0
    for index in range(all_sim.shape[0]):
        if Y_train[np.argmax(all_sim[index])] == Y_train[index]:
            right += 1
        else:
            wrong += 1
    print 'right', right, 'wrong', wrong, 'acc', (right * 1.0 / (wrong+right))



if __name__ == '__main__':
    X_train_feature, Y_train, X_test_feature, Y_test = \
        msgpack_numpy.load(open('sim_threshold_feature.p', 'rb'))
    cal_sim_threshold(X_train_feature, Y_train)

