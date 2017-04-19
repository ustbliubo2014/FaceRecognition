# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/11/1 19:52
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LassoLars
from sklearn.linear_model import SGDClassifier
import pdb

#regressionFunc =LogisticRegression(C=10, penalty='l1', tol=0.0001)

RandomForest=RandomForestClassifier(n_estimators=10)
file_train="C:\Users\liubo\Desktop\lr_train/all_feature_label_20161029_train"
file_test="C:\Users\liubo\Desktop\lr_train/all_feature_label_20161029_test"
train_x, train_y = load_svmlight_file(file_train)
test_x, test_y = load_svmlight_file(file_test)

#test_x, test_y = load_svmlight_file(file_test)
#svm=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
#svm.fit(train_x,train_y)
#test_sco=svm.score(test_x,test_y)
RandomForest.fit(train_x,train_y)
test_sco=RandomForest.score(test_x,test_y)
print(test_sco)
regressionFunc = LogisticRegression(penalty='l1', dual=False, tol=1e-4, C=10.0, solver ='liblinear')
train_sco=regressionFunc.fit(train_x,train_y).score(train_x,train_y)
print(train_sco)
test_sco=regressionFunc.score(test_x,test_y)
print(test_sco)
