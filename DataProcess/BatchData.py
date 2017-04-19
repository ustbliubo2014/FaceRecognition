# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: BatchData.py
@time: 2016/7/15 10:42
@contact: ustb_liubo@qq.com
@annotation: BatchData : 批量读入数据,放入队列
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import logging
from Data import ImageData
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='BatchData.log',
                    filemode='a+')


class BatchData(ImageData):
    def __init__(self, init_args, *args, **kwargs):
        ImageData.__init__(self, init_args)




if __name__ == '__main__':
    pass
