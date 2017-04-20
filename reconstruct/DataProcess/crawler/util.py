# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: util.py
@time: 2016/7/28 11:38
@contact: ustb_liubo@qq.com
@annotation: util
"""
import sys
import logging
from logging.config import fileConfig
import os
import requests
from bs4 import BeautifulSoup
import json
reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


def search_360_index(person_name):
    # 也是使用ajax的方法
    try:
        url = 'http://index.so.com/index.php?a=overviewJson' \
              '&q=%s&area=全国' % person_name
        response = requests.get(url)
        html_doc = response.content
        soup = BeautifulSoup(html_doc)
        dic = json.loads(soup.text)
        return dic.get('data')[0].get('data').get('month_index')
    except:
        return 0


def sort_person(file_name):
    # 根据搜索指数排序
    person_list = []
    for line in open(file_name):
        tmp = line.rstrip().split('\t')
        if len(tmp) == 2:
            person_name = tmp[0]
            search_index = int(tmp[1])
            person_list.append((person_name, search_index))
    person_list.sort(key=lambda x: x[1], reverse=True)
    f = open(file_name, 'w')
    for person_name, search_index in person_list:
        f.write(str(person_name)+'\t'+str(search_index)+'\n')
    f.close()


if __name__ == '__main__':
    # print search_360_index('qiaodan')
    # print search_360_index('乔丹')
    pass
