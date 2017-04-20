# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: crawler_person.py
@time: 2016/7/28 11:38
@contact: ustb_liubo@qq.com
@annotation: crawler_person
"""
import sys
import logging
from logging.config import fileConfig
import os
from util import search_360_index, sort_person
from conf import headers
import requests
import json
from bs4 import BeautifulSoup
import traceback
import pdb

reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


class PersonCrawler():
    '''
        从网站爬取人名列表
    '''
    def __init__(self, result_file, host):
        '''
            :param result_file: 最后人名存放的文件名
            :param host: 网站host
            :return:
        '''
        self.result_file = result_file
        self.host = host
        self.headers = headers
        self.headers['host'] = self.host
        self.file = open(result_file, 'a')

    def get_url_list(self):
        # 根据网站规则, 获取人名列表(不同网站方法不同,需要在子类中实现)
        pass

    def get_content(self, url):
        response = requests.get(url, headers=headers)
        status = response.status_code
        if status == 200:
            return response.content
        else:
            return None

    def analyse_content(self, content, has_find_dic):
        # 从content中解析出人名, 在子类中实现
        pass

    def crawler(self):
        # 主程序
        pass

    def sort(self):
        sort_person(self.result_file)


class Movie1905(PersonCrawler):
    def __init__(self, result_file, host):
        PersonCrawler.__init__(self, result_file, host)

    def get_url_list(self):
        page_num = 1837
        url_list = []
        for index in range(1, page_num):
            url = 'http://www.1905.com/mdb/star/m1p%d.html' % index
            url_list.append(url)
        return url_list

    def analyse_content(self, content, has_find_dic):
        soup = BeautifulSoup(content)
        try_num = 10
        while try_num > 0:
            this_person_list = soup.findAll(attrs={'class': 'ta_c mt05'})
            try_num -= 1
            if len(this_person_list) > 0:
                for element in this_person_list:
                    try:
                        person_name = element.text
                        if person_name in has_find_dic:
                            continue
                        search_index = search_360_index(person_name)
                        self.file.write(
                            str(person_name)+'\t'+str(search_index)+'\n')
                        has_find_dic[person_name] = search_index
                    except:
                        traceback.print_exc()
                        continue
                break

    def crawler(self):
        url_list = self.get_url_list()
        has_find_dic = {}
        for url in url_list:
            try:
                content = self.get_content(url)
                self.analyse_content(content, has_find_dic)
            except:
                continue
        self.file.close()


if __name__ == '__main__':
    movie1905_host = 'www.1905.com'
    movie1905_file = 'movie1905_person.txt'
    movie1905Crawler = Movie1905(movie1905_file, movie1905_host)
    # movie1905Crawler.crawler()
    movie1905Crawler.sort()
