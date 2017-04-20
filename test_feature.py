# -*-coding:utf-8 -*-
__author__ = 'liubo-it'

# 处理test提取特质之后的数据 (用分类器训练模型; 训练阈值)
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
import msgpack_numpy
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    feature_file = '/data/liubo/face/vgg_face_dataset/test_deepid_feature.p'
    (feature_data, feature_label, label_trans_dic) = msgpack_numpy.load(open(feature_file,'rb'))
    for index in range(5):
        rf = RandomForestClassifier(n_estimators=500, n_jobs=15)
        X_train, X_test, y_train, y_test = train_test_split(feature_data, feature_label, test_size=0.3, random_state=0)
        rf.fit(X_train, y_train)
        print 'rf test acc : ', accuracy_score(y_test, rf.predict(X_test))
        print 'rf train acc : ', accuracy_score(y_train, rf.predict(X_train))
        n_neighbors = 10
        for weights in ['uniform', 'distance']:
            knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            knn.fit(X_train, y_train)
            print 'knn %s test acc : ' %weights, accuracy_score(y_test, knn.predict(X_test))
            print 'knn %s train acc : ' %weights, accuracy_score(y_train, knn.predict(X_train))





