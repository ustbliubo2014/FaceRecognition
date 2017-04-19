# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: pic_crawler.py
@time: 2016/7/26 15:49
@contact: ustb_liubo@qq.com
@annotation: pic_crawler
"""
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='pic_crawler.log',
                    filemode='a+')
from logging.config import fileConfig

fileConfig('logger_config.ini')
logger_info = logging.getLogger('infohandler')



import urllib2
from bs4 import BeautifulSoup
from time import time, sleep
import os
import base64


def crawler_img(person_name, limit):
    pic_url_list = set()
    try:
        for page_index in range(10):
            try:
                url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=%s&pn=%d' % (person_name, page_index*20)
                req = urllib2.Request(url, None)
                response = urllib2.urlopen(req)
                html_doc = response.read()
                soup = BeautifulSoup(html_doc)
                str_soup = str(soup)
                last_index = 0
                while True:
                    start_index = str_soup.find('objURL', last_index)
                    if start_index != -1:#还有objURL
                        end_index = str_soup.find(',', start_index)
                        if end_index != -1:
                            pic_url = str_soup[start_index:end_index][9:-1]
                            pic_url_list.add(pic_url)
                            last_index = end_index
                    else:
                        break
            except:
                continue
            if len(pic_url_list) > limit:
                break
    except:
        return pic_url_list
    return pic_url_list


def crawler_all_person(person_file, person_url_folder):
    '''
        :param person_file: 人名列表
        :param person_url_folder: url文件列表
        :return:
    '''
    crawler_num_limit = 3700
    pic_num_limit = 100
    if not os.path.exists(person_url_folder):
        os.makedirs(person_url_folder)
    count = 0
    for line in open(person_file):
        try:
            count += 1
            logger_info.info(str(count))
            if count > crawler_num_limit:
                break
            tmp = line.rstrip().split('\t')
            if len(tmp) == 2:
                person_name = tmp[0].encode('utf-8').decode('utf-8')
                url_file = os.path.join(person_url_folder, person_name+'.txt')
                # 已经爬过这个人的url了
                if os.path.exists(url_file):
                    continue
                pic_url_list = crawler_img(person_name, pic_num_limit)
                with open(url_file, 'w') as f:
                    for index, pic_url in enumerate(pic_url_list):
                        f.write(person_name+'\t'+str(index)+'\t'+pic_url+'\n')
                    f.close()
            sleep(10)
        except:
            continue


def pic_download(url, person_name, file_name):
    try:
        a = urllib2.urlopen(url)
        pic = base64.b64encode(a.read())
        print '\t'.join([person_name, file_name, pic])
    except:
        return


def crawler_map():
    for line in sys.stdin:
        try:
            tmp = line.rstrip().split()
            if len(tmp) >= 3:
                person_name = tmp[0]
                file_name = tmp[1]
                url = tmp[2]
                pic_download(url, person_name, file_name)
        except:
            continue


if __name__ == '__main__':
    person_file = u'person_list.txt'
    person_url_folder = 'url'
    crawler_all_person(person_file, person_url_folder)
    #
    # crawler_map()
