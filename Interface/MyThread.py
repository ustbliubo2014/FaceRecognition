# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: MyThread.py
@time: 2016/6/14 17:12
@contact: ustb_liubo@qq.com
@annotation: MyThread
"""
import sys
import threading
from time import ctime
reload(sys)
sys.setdefaultencoding("utf-8")


def func():
    pass


class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    def getResult(self):
        return self.res

    def run(self):
        print 'starting', self.name, 'at:',ctime()
        self.res=apply(self.func,self.args)
        print self.name, 'finished at:', ctime()
if __name__ == '__main__':
    pass
