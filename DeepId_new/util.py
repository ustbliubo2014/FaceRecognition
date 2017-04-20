# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: util.py
@time: 2016/7/4 16:24
@contact: ustb_liubo@qq.com
@annotation: util
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
import numpy as np


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='util.log',
                    filemode='w')


def writer(queue, epoch_num, batch_size, sample_list, batch_num, person_num, pic_shape,
           load_data_func, pic_func):
    '''
        :param queue: 共享队列
        :param epoch_num:
        :param batch_size:
        :param sample_list:样本位置(给出位置,直接读取)
        :param batch_num: len(sample_list) / batch_size  --- 在train中也要知道读多少数据,需要给出epoch_num*batch_num
        :param load_data_func: 读取哪种类型的数据 [load_rgb_batch_data, load_gray_batch_data]
        :param pic_func: 对每个图片进行处理
        :return:
    '''

    for epoch_id in range(epoch_num):
        np.random.shuffle(sample_list)
        for batch_id in range(batch_num):
            batch_x, batch_y = \
                load_data_func(sample_list[batch_id*batch_size:(batch_id+1)*batch_size], person_num,
                               pic_shape, pic_func)
            queue.put((batch_x, batch_y))



if __name__ == '__main__':
    pass
