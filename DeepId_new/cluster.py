# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: cluster.py
@time: 2016/7/4 17:17
@contact: ustb_liubo@qq.com
@annotation: cluster: 多个文件进行聚类
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='cluster.log',
                    filemode='w')

if __name__ == '__main__':
    pass
