# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: experiment.py
@time: 2016/11/24 14:56
@contact: ustb_liubo@qq.com
@annotation: experiment
"""
import sys
sys.path.append('/home/liubo-it/FaceRecognization/')
import os
from Interface.research_model import extract_feature_from_file
# from Interface.light_cnn_model import extract_feature_from_file as self_extract_feature_from_file
import msgpack_numpy
import base64
from time import time
from sklearn.neighbors import LSHForest
import numpy as np
import pdb
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from keras.models import model_from_json
import keras.backend as K
from sklearn.decomposition import PCA

reload(sys)
sys.setdefaultencoding("utf-8")

# sim_threshold = 0.65
sim_threshold = 0.8
max_person_num = 20


def extract_all_feature(folder, result_file, extract_func):
    f = open(result_file, 'w')
    f.write('pic_path'+'\t'+'feature'+'\n')
    person_list = os.listdir(folder)
    for person in person_list:
        print person
        person_path = os.path.join(folder, person)
        pic_list = os.listdir(person_path)
        for pic in pic_list:
            pic_path = os.path.join(person_path, pic)
            # feature = extract_feature_from_file(pic_path)
            feature = extract_func(pic_path)
            feature_str = base64.b64encode(msgpack_numpy.dumps(feature))
            f.write(pic_path+'\t'+feature_str+'\n')
    f.close()


def load_data(result_file, pack_file):
    person_feature_dic = {} # {person_name:[(pic_name, pic_feature),...,(pic_name, pic_feature)]}
    for line in open(result_file):
        tmp = line.rstrip().split('\t')
        if len(tmp) == 2:
            try:
                pic_path = tmp[0].split('/')
                person_name = pic_path[-2]
                pic_name = pic_path[-1]
                feature = msgpack_numpy.loads(base64.b64decode(tmp[1]))
                feature_list = person_feature_dic.get(person_name, [])
                feature_list.append((pic_name, feature))
                person_feature_dic[person_name] = feature_list
            except:
                print tmp
                continue
    msgpack_numpy.dump(person_feature_dic, open(pack_file, 'wb'))


def split_train_valid(pack_file, train_pic_num, feature_dim):
    start = time()
    person_feature_dic = msgpack_numpy.load(open(pack_file, 'rb'))
    all_train_data = []
    all_train_label = []
    all_valid_data = []
    all_valid_label = []
    for person_index, person in enumerate(person_feature_dic):
        feature_list = person_feature_dic.get(person)
        np.random.shuffle(feature_list)
        if len(feature_list) < train_pic_num:
            continue
        else:
            for index in range(train_pic_num):
                pic_name, feature = feature_list[index]
                feature = np.asarray(feature)
                if feature.shape != (1, feature_dim):
                    continue
                all_train_data.append(feature)
                all_train_label.append(person)
            for index in range(train_pic_num, len(feature_list)):
                pic_name, feature = feature_list[index]
                feature = np.asarray(feature)
                if feature.shape != (1, feature_dim):
                    continue
                all_valid_data.append(feature)
                all_valid_label.append(person)
    all_train_data = np.asarray(all_train_data)
    all_train_label = np.asarray(all_train_label)
    all_valid_data = np.asarray(all_valid_data)
    all_valid_label = np.asarray(all_valid_label)
    return all_train_data, all_train_label, all_valid_data, all_valid_label


def cal_acc(pack_file, stat_file, feature_dim):
    f = open(stat_file, 'w')
    f.write('train_pic_num'+'\t'+'person_name'+'\t'+'acc'+'\n')
    pic_num = range(1, max_person_num)
    for num in pic_num:
        all_train_data, all_train_label, all_valid_data, all_valid_label = split_train_valid(pack_file, train_pic_num=num, feature_dim=feature_dim)
        lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=5)

        for index in range(len(all_train_data)):
            try:
                if all_train_data[index] == None:
                    continue
                lshf.partial_fit(all_train_data[index], all_train_label[index])
            except:
                traceback.print_exc()
                continue
        # 对于每个人,分别统计准确率
        person_acc_dic = {}     # 准确的个数
        person_all_dic = {}     # 总的个数
        filter_num = 0
        all_num = 0
        for index in range(len(all_valid_data)):
            try:
                if all_valid_data[index] == None:
                    continue
                all_find_distance, all_find_index = lshf.kneighbors(all_valid_data[index], n_neighbors=5, return_distance=True)
                cos_sim = cosine_similarity(all_valid_data[index], all_train_data[all_find_index[0, 0]])
                label = all_train_label[all_find_index[0, 0]]
                # if cos_sim > sim_threshold:
                if True:
                    if label == all_valid_label[index]:
                        person_acc_dic[label] = person_acc_dic.get(label, 0) + 1
                        person_all_dic[label] = person_all_dic.get(label, 0) + 1
                    else:
                        person_all_dic[label] = person_all_dic.get(label, 0) + 1
                else:
                    filter_num += 1
                all_num += 1
            except:
                print all_valid_label[index]
                continue
        print 'train_num :', num, 'filter_rate: ', (filter_num * 1.0 / all_num)
        for person in person_all_dic:
            all_num = person_all_dic[person]
            right_num = person_acc_dic.get(person, 0)
            f.write('\t'.join(map(str, [num, person, (right_num * 1.0 /  all_num)]))+'\n')


def cal_recall(pack_file, stat_file, feature_dim):
    # f_model = open('verf.txt', 'w')
    f = open(stat_file, 'w')
    f.write('train_pic_num'+'\t'+'person_name'+'\t'+'recall'+'\n')
    pic_num = range(1, max_person_num)
    for num in pic_num:
        all_train_data, all_train_label, all_valid_data, all_valid_label = split_train_valid(pack_file, train_pic_num=num, feature_dim=feature_dim)
        lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=5)
        for index in range(len(all_train_data)):
            try:
                if all_train_data[index] == None:
                    continue
                lshf.partial_fit(all_train_data[index], all_train_label[index])
            except:
                continue
        # 对于每个人,分别统计准确率
        person_find_dic = {}     # 准确的个数
        person_all_dic = {}     # 总的个数
        for index in range(len(all_valid_data)):
            try:
                if all_valid_data[index] == None:
                    continue
                all_find_distance, all_find_index = lshf.kneighbors(all_valid_data[index], n_neighbors=5, return_distance=True)
                cos_sim = cosine_similarity(all_valid_data[index], all_train_data[all_find_index[0, 0]])
                label = all_train_label[all_find_index[0, 0]]
                real_label = all_valid_label[index]
                # if cos_sim > sim_threshold:
                if True:
                    if label == real_label:
                        # f_model.write('0'+'\t'+str(cos_sim)+'\n')
                        person_find_dic[real_label] = person_find_dic.get(real_label, 0) + 1
                        person_all_dic[real_label] = person_all_dic.get(real_label, 0) + 1
                    else:
                        # f_model.write('1' + '\t' + str(cos_sim) + '\n')
                        person_all_dic[real_label] = person_all_dic.get(real_label, 0) + 1
            except:
                print all_valid_label[index]
                continue
        print 'train_num :', num
        for person in person_all_dic:
            all_num = person_all_dic[person]
            right_num = person_find_dic.get(person, 0)
            f.write('\t'.join(map(str, [num, person, (right_num * 1.0 /  all_num)]))+'\n')


def show_result(stat_file, show_file):
    person_dic = {}
    for line in open(stat_file):
        if line.startswith('train_pic_num'):
            continue
        tmp = line.rstrip().split('\t')
        if len(tmp) == 3:
            train_pic_num, person_name, acc = tmp
            acc_list =  person_dic.get(person_name, [])
            acc_list.append(acc)
            person_dic[person_name] = acc_list
    f = open(show_file, 'w')
    for person in person_dic:
        print person, len(person_dic.get(person))
        f.write(person+'\t'+'\t'.join(person_dic.get(person))+'\n')
    f.close()


def feature_trans_autoencoder(src_pack_file, dst_pack_file):
    weight_file = '/data/liubo/face/annotate_face_model/skyeye_face_autoencoder.weight'
    model_file = '/data/liubo/face/annotate_face_model/skyeye_face_autoencoder.model'
    autoencoder =  model_from_json(open(model_file, 'r').read())
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.load_weights(weight_file)
    get_Conv_FeatureMap = K.function([autoencoder.layers[0].get_input_at(False), K.learning_phase()],
                                     [autoencoder.layers[-2].get_output_at(False)])

    person_feature_dic = msgpack_numpy.load(open(src_pack_file, 'rb'))
    for person_index, person in enumerate(person_feature_dic):
        feature_list = person_feature_dic.get(person)
        for index in range(len(feature_list)):
            try:
                if feature_list[index][1] == None:
                    continue
                this_feature = np.array(feature_list[index][1][0])
                this_feature = np.reshape(this_feature, (1, this_feature.size))
                this_feature = get_Conv_FeatureMap([this_feature, 0])[0][0]
                feature_list[index][1][0] = this_feature
            except:
                traceback.print_exc()
    msgpack_numpy.dump(person_feature_dic, open(dst_pack_file, 'wb'))


def feature_trans_pca(src_pack_file, dst_pack_file):
    all_data = []

    person_feature_dic = msgpack_numpy.load(open(src_pack_file, 'rb'))
    for person_index, person in enumerate(person_feature_dic):
        feature_list = person_feature_dic.get(person)
        for index in range(len(feature_list)):
            try:
                if feature_list[index][1] == None:
                    continue
                this_feature = np.array(feature_list[index][1][0])
                all_data.append(this_feature)
            except:
                traceback.print_exc()
    all_data = np.asarray(all_data)
    pca = PCA(n_components=128)
    pca.fit(all_data)

    for person_index, person in enumerate(person_feature_dic):
        feature_list = person_feature_dic.get(person)
        for index in range(len(feature_list)):
            try:
                if feature_list[index][1] == None:
                    continue
                this_feature = np.array(feature_list[index][1][0])
                this_feature = np.reshape(this_feature, (1, this_feature.size))
                this_feature = pca.transform(this_feature)[0]
                feature_list[index][1][0] = this_feature
            except:
                traceback.print_exc()
    msgpack_numpy.dump(person_feature_dic, open(dst_pack_file, 'wb'))



def test_research():
    # feature_dim = 256
    # pack_file = 'person_feature_dic_research.p'
    # stat_acc_file = 'accuracy_stat_research.txt'
    # stat_recall_file = 'recall_stat_research.txt'
    # show_acc_file = 'accuracy_show_research.txt'
    # show_recall_file = 'recall_show_research.txt'

    # feature_dim = 128
    # pack_file = 'autoencoder_person_feature_dic_research.p'
    # stat_acc_file = 'accuracy_stat_research_autoencoder.txt'
    # stat_recall_file = 'recall_stat_research_autoencoder.txt'
    # show_acc_file = 'accuracy_show_research_autoencoder.txt'
    # show_recall_file = 'recall_show_research_autoencoder.txt'

    feature_dim = 128
    pack_file = 'pca_person_feature_dic_research.p'
    stat_acc_file = 'accuracy_stat_research_pca.txt'
    stat_recall_file = 'recall_stat_research_pca.txt'
    show_acc_file = 'accuracy_show_research_pca.txt'
    show_recall_file = 'recall_show_research_pca.txt'

    # load_data(result_file='/data/liubo/face/research_person_face_research.txt', pack_file='person_feature_dic2.p')
    cal_acc(pack_file=pack_file, stat_file=stat_acc_file, feature_dim=feature_dim)
    show_result(stat_file=stat_acc_file, show_file=show_acc_file)
    cal_recall(pack_file=pack_file, stat_file=stat_recall_file, feature_dim=feature_dim)
    show_result(stat_file=stat_recall_file, show_file=show_recall_file)


def test_skyeye():
    feature_dim = 512
    pack_file = 'person_feature_dic_self.p'
    stat_acc_file = 'accuracy_stat_self.txt'
    stat_recall_file = 'recall_stat_self.txt'
    show_acc_file = 'accuracy_show_self.txt'
    show_recall_file = 'recall_show_self.txt'
    # load_data(result_file='/data/liubo/face/research_person_face_research.txt', pack_file='person_feature_dic2.p')
    cal_acc(pack_file=pack_file, stat_file=stat_acc_file, feature_dim=feature_dim)
    cal_recall(pack_file=pack_file, stat_file=stat_recall_file, feature_dim=feature_dim)
    show_result(stat_file=stat_acc_file, show_file=show_acc_file)
    show_result(stat_file=stat_recall_file, show_file=show_recall_file)


if __name__ == '__main__':
    pass
    # extract_all_feature(folder='/data/liubo/face/research_person_face2/research_person_face',
    #                     result_file='/data/liubo/face/research_person_face_research.txt', extract_func=extract_feature_from_file)

    test_research()
    # test_skyeye()
    # feature_trans_autoencoder(src_pack_file='person_feature_dic_research.p', dst_pack_file='autoencoder_person_feature_dic_research.p')
    # feature_trans_pca(src_pack_file='person_feature_dic_research.p', dst_pack_file='pca_person_feature_dic_research.p')

