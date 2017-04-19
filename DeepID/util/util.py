# -*-coding:utf-8 -*-
__author__ = 'liubo-it'

import pdb

# top-5准确率
def get_top5_acc(y_prob, y_true):
    acc_num = 0
    for index,y in enumerate(y_true):
        this_y_prob = y_prob[index]
        tmp = sorted(zip(this_y_prob,range(len(this_y_prob))),key=lambda x:x[0], reverse=True)
        for k in range(5):
            if y == tmp[k][1]:
                acc_num += 1
    return acc_num*1.0/len(y_true)

