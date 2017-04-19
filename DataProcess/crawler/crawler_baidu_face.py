# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: crawler_baidu_face.py
@time: 2016/7/29 18:05
@contact: ustb_liubo@qq.com
@annotation: crawler_baidu_face : 爬取百度人脸识别的接口
"""
import sys
import logging
from logging.config import fileConfig
import os
from bs4 import BeautifulSoup
import urllib
import requests
import pdb
import traceback
from conf import *
from time import time, sleep
from Analog_Internet_access import BrowserBase


reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


https_proxy = '183.60.218.67:8360'
proxyDict = {'https': https_proxy}

headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
               "Accept-Encoding": "gzip, deflate, sdch",
               "Accept-Language": "zh-CN,zh;q=0.8",
               "Cache-Control": "max-age=0",
               "Proxy-Connection": "keep-alive",
               "Upgrade-Insecure-Requests": 1,
               "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/50.0.2661.102 Safari/537.36"}


def get_baidu_url(pic_url):
    pic_url = urllib.quote(pic_url.encode('utf-8', 'replace')).replace('/', '%2F')
    baidu_url = 'http://image.baidu.com/n/pc_search?' \
                'queryImageUrl=%s&querySign=&simid=&' \
                'fm=index&pos=&uptype=paste' % pic_url
    return baidu_url


def analyse(pic_url):
    # 文件夹的名字也要和找到的两个词条相同
    newbaike_name = ''
    newbaike_similarity = '0'
    guess_info = ''
    try:
        try:
            baidu_url = get_baidu_url(pic_url).rstrip()
            # print 'baidu_url :', baidu_url
        except:
            newbaike_name = url_error1_str
            guess_info = url_error2_str
            return newbaike_name.rstrip(), newbaike_similarity.rstrip(), guess_info.rstrip(), baidu_url
        try:
            try_num = 1
            has_find = False
            while try_num > 0:
                response = requests.get(url=baidu_url,
                                        proxies=proxyDict,
                                        timeout=30,
                                        headers=headers
                )
                status_code = response.status_code
                # print 'status_code :', status_code
                if response.status_code == 200:
                    has_find = True
                    break
                try_num -= 1
            if not has_find:
                newbaike_name = timeout1_str
                guess_info = timeout2_str
                newbaike_similarity = str(status_code)
                return newbaike_name.rstrip(), newbaike_similarity.rstrip(), guess_info.rstrip(), baidu_url
        except:
            newbaike_name = timeout1_str
            guess_info = timeout2_str
            return newbaike_name.rstrip(), newbaike_similarity.rstrip(), guess_info.rstrip(), baidu_url
        try:
            soup = BeautifulSoup(response.content)
            try:
                try_num = 1
                has_find = False
                while try_num > 0:
                    try_num -= 1
                    newbaike_name_list = soup.select('.guess-newbaike-name')
                    newbaike_similarity_list = soup.select('.guess-newbaike-left-similarity')
                    if len(newbaike_name_list) > 0 and len(newbaike_similarity_list) > 0:
                        newbaike_name = newbaike_name_list[0].text
                        newbaike_similarity = newbaike_similarity_list[0].text
                        has_find = True
                        break
                if not has_find:
                    newbaike_name = no_newbaike_name
                    newbaike_similarity = '0'
            except:
                newbaike_name = no_newbaike_name
                newbaike_similarity = '0'
            try:
                try_num = 1
                has_find = False
                while try_num > 0:
                    guess_info_list = soup.select('.guess-info-text')
                    if len(guess_info_list) > 0:
                        guess_info = guess_info_list[0].text
                        has_find = True
                        break
                if not has_find:
                    guess_info = no_guess_info
            except:
                guess_info = no_guess_info
        except:
            newbaike_name = analyse_error1_str
            guess_info = analyse_error2_str
            return newbaike_name.rstrip(), newbaike_similarity.rstrip(), guess_info.rstrip(), baidu_url
        return newbaike_name.rstrip(), newbaike_similarity.rstrip(), guess_info.rstrip(), baidu_url
    except:
        return newbaike_name.rstrip(), newbaike_similarity.rstrip(), guess_info.rstrip(), baidu_url


def crawler_all_url_hadoop():
    count = 0
    for line in sys.stdin:
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            name = tmp[0]
            index = tmp[1]
            pic_url = tmp[2]
            newbaike_name, newbaike_similarity, guess_info, baidu_url \
                    = analyse(pic_url)
            if newbaike_similarity == '403':
                raise EOFError
            content = [
                    name, index, pic_url, newbaike_name,
                    newbaike_similarity, guess_info, baidu_url]
            print '\t'.join(map(str, content)).replace('\n', '')


def crawler_one_file(file_path):
    count = 0
    for line in open(file_path):
        count += 1
        tmp = line.rstrip().split()
        if len(tmp) == 3:
            name = tmp[0]
            index = tmp[1]
            pic_url = tmp[2]
            newbaike_name, newbaike_similarity, guess_info \
                    = analyse(pic_url)
            if newbaike_similarity == '403':
                raise EOFError
            content = [
                    name, index, pic_url, newbaike_name,
                    newbaike_similarity, guess_info]
            print '\t'.join(map(str, content)).replace('\n', '')


if __name__ == '__main__':
    crawler_all_url_hadoop()

    # tmp = analyse('http://imgsrc.baidu.com/forum/w%3D580/sign=7cdc14009f16fdfad86cc6e6848f8cea/579cecfaaf51f3de6fbc564f96eef01f3a29791e.jpg')
    # print tmp
