# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: DeepId_batch.py
@time: 2016/7/4 16:23
@contact: ustb_liubo@qq.com
@annotation: DeepId_batch
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging
from Queue import Queue
import msgpack
from MyThread import MyThread
from util import writer
from DeepId_batch_model import train
from load_data import load_rgb_batch_data, load_gray_batch_data
from critical_point_detect import get_nose, get_left_eye, get_right_eye
from load_data import flip_lr, rotate_img


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='DeepId_batch.log',
                    filemode='w')



class DeepIdBatch(object):
    def __init__(self, pic_type, model_type):
        maxsize = 20
        batch_size = 128
        epoch_num = 10
        self.train_queue = Queue(maxsize)
        self.valid_queue = Queue(maxsize)
        self.sample_list_file = '/data/liubo/face/vgg_face_dataset/all_data/all_sample_list.p'
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        func_args_dic = {flip_lr: (), rotate_img: (30,)}
        self.data_folder = '/data/liubo/face/vgg_face_dataset/'
        if pic_type == 'rgb':
            self.pic_shape = (50, 50, 3)
            self.input_shape = (3, 50, 50)
            self.person_num = None
            self.load_func = load_rgb_batch_data
            if model_type == 'whole':
                person_num = None
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.model'
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.deepid.weight'
            elif model_type == 'rgb_right':
                func_args_dic[get_right_eye] = (self.pic_shape, )
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.right_eye.deepid.weight'
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.right_eye.deepid.model'
            elif model_type == 'rgb_left':
                func_args_dic[get_left_eye] = (self.pic_shape, )
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.left_eye.deepid.weight'
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.small.rgb.left_eye.deepid.model'
            elif model_type == 'rgb_nose':
                func_args_dic[get_nose] = (self.pic_shape, )
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.weight'
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.all.rgb.nose.deepid.model'
            else:
                print 'error model type'
                sys.exit()
            self.pic_func = func_args_dic
        elif pic_type == 'gray':
            self.load_func = load_gray_batch_data
            if model_type == 'nose':
                func_args_dic[get_nose] = (self.pic_shape)
                self.input_shape = (1, 88, 128)# 每个图片的shape
                self.pic_shape = (88, 128)
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.nose.model'
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.nose.weight'

            elif model_type == 'left_eye':
                func_args_dic[get_left_eye] = (self.pic_shape, )
                self.input_shape = (1, 128, 88)# 每个图片的shape
                self.pic_shape = (128, 88)
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.left_eye.model'
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.left_eye.weight'

            elif model_type == 'right_eye':
                func_args_dic[get_right_eye] = (self.pic_shape, )
                self.input_shape = (1, 128, 88)# 每个图片的shape
                self.pic_shape = (128, 88)
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.right_eye.model'
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.right_eye.weight'

            elif model_type == 'whole':
                self.input_shape = (1, 128, 88)# 每个图片的shape
                self.pic_shape = (128, 88)
                self.model_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.model'
                self.weight_file = '/data/liubo/face/vgg_face_dataset/model/vgg_face.all_data.big.gray.deepid.weight'
            else:
                print 'error param'
                sys.exit()
            self.pic_func = func_args_dic
        else:
            print 'error model type'
            sys.exit()


    def read_train(self):
        train_sample_list, valid_sample_list = msgpack.load(open(self.sample_list_file, 'rb'))
        person_num = len(set([tmp[1] for tmp in train_sample_list]))
        train_batch_num = len(train_sample_list) / self.batch_size
        valid_batch_num = len(valid_sample_list) / self.batch_size
        train_valid_model_args = (self.input_shape, person_num, self.model_file, self.weight_file, self.train_queue,
                    self.valid_queue, self.epoch_num, train_batch_num, valid_batch_num,)
        train_write_args = (self.train_queue, self.epoch_num, self.batch_size, train_sample_list,
                            train_batch_num, person_num, self.pic_shape, self.load_func, self.pic_func)
        valid_write_args = (self.valid_queue, self.epoch_num, self.batch_size, valid_sample_list,
                            valid_batch_num, person_num, self.pic_shape, self.load_func, self.pic_func)
        train_valid_model_thread = MyThread(func=train, args=train_valid_model_args, name='train_valid_model')
        train_write_thread = MyThread(func=writer, args=train_write_args, name='train_write')
        valid_write_thread = MyThread(func=writer, args=valid_write_args, name='valid_write')
        train_valid_model_thread.start()
        train_write_thread.start()
        valid_write_thread.start()
        train_valid_model_thread.join()
        train_write_thread.join()
        valid_write_thread.join()


if __name__ == '__main__':
    deepid_batch = DeepIdBatch(pic_type='rgb', model_type=sys.argv[1])
    deepid_batch.read_train()
