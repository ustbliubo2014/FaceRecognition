# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: crawler.py
@time: 2016/7/21 12:07
@contact: ustb_liubo@qq.com
@annotation: crawler
"""
import sys
import traceback
import logging
from time import time, sleep
import json
import urllib2
from bs4 import BeautifulSoup
import pdb
import msgpack
import re
import json
import requests
reload(sys)
sys.setdefaultencoding("utf-8")


def crawler_baidu_person_list():
    # 从百度人气榜上获取人名
    all_person_list = set()
    for index in range(50):
        url = 'http://baike.baidu.com/operation/api/starflowerstarlist?' \
              'rankType=thisWeek&pg=%d' % index
        req = urllib2.Request(url, None)
        response = urllib2.urlopen(req)
        html_doc = response.read()
        content = json.loads(html_doc)
        this_page_list = content.get('data').get('thisWeek')
        for person_content in this_page_list:
            all_person_list.add(person_content.get('name'))
    print len(all_person_list)

    for index in range(50):
        url = 'http://baike.baidu.com/operation/api/starflowerstarlist?' \
              'rankType=lastWeek&pg=%d' % index
        req = urllib2.Request(url, None)
        response = urllib2.urlopen(req)
        html_doc = response.read()
        content = json.loads(html_doc)
        this_page_list = content.get('data').get('lastWeek')
        for person_content in this_page_list:
            all_person_list.add(person_content.get('name'))
    print len(all_person_list)
    all_person_set = list(all_person_list)
    msgpack.dump(all_person_set, open('baidu_fans.p', 'w'))


def crawler_fans():
    all_person_set = set()
    f_fans = open('fans.txt', 'w')
    for page_index in range(1, 345, 1):
        for class_index in [1, 2, 4, 5, 6]:
            start = time()
            try:
                url = 'https://123fans.cn/results.php?qi=%d&c=%d'\
                      % (page_index, class_index)
                req = urllib2.Request(url, None)
                response = urllib2.urlopen(req)
                html_doc = response.read()
                soup = BeautifulSoup(html_doc)
                children = [k for k in soup.children][1]
                lis = children.select('.odd')
                for k in lis:
                    try:
                        name = k.select('.name')[0].string
                        all_person_set.add(name)
                        f_fans.write(name+'\n')
                    except:
                        traceback.print_exc()
                        continue
                print page_index, class_index, \
                    len(all_person_set), time()-start
            except:
                print 'error', page_index, class_index
        sleep(5)
    all_person_set = list(all_person_set)
    msgpack.dump(all_person_set, open('fans.p', 'w'))


def crawler_baidu_online():
    url = 'http://www.zwbk.org/MyLemmaType.aspx?tid=001016'
    req = urllib2.Request(url, None)
    response = urllib2.urlopen(req)
    html_doc = response.read()
    soup = BeautifulSoup(html_doc)
    url_list = soup.find_all(href=re.compile("MyTypeLayerSecond"))
    part_url_list = []
    for index in range(len(url_list)):
        part_url_list.append(url_list[index]['href'])
    f = open(u'中文百科在线.txt', 'w')
    person_list = []
    for part_url in part_url_list:
        real_part_url = 'http://www.zwbk.org/'+part_url
        part_req = urllib2.Request(real_part_url, None)
        part_response = urllib2.urlopen(part_req)
        part_html_doc = part_response.read()
        part_soup = BeautifulSoup(part_html_doc)
        tmp = part_soup.select('.sdiv2')
        for k in tmp:
            person_name = k.text.strip()
            search_index = int(search_360_index(person_name))
            person_list.append((person_name, search_index))
        print real_part_url, len(person_list)
    person_list.sort(key=lambda x: x[1], reverse=True)
    for person_name, search_index in person_list:
        f.write(str(person_name)+'\t'+str(search_index)+'\n')
    f.close()


def crawler_1905_movie():

    headers = {"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
               "Accept-Encoding":"gzip, deflate, sdch","Accept-Language":"zh-CN,zh;q=0.8",
               "Cache-Control":"max-age=0","Host":"www.1905.com","Proxy-Connection":"keep-alive",
               "Upgrade-Insecure-Requests":1,
               "User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/50.0.2661.102 Safari/537.36"}
    fail_url_dic = {}
    page_num = 1837
    all_person_set = set()
    f = open('1905_movie.txt', 'w')
    for index in range(1, page_num):
        url = 'http://www.1905.com/mdb/star/m1p%d.html' % index
        response = requests.get(url, headers=headers)
        status = response.status_code
        content = response.content
        soup = BeautifulSoup(content)
        try_num = 10
        not_find = True
        while try_num > 0:
            this_person_list = soup.findAll(attrs={'class': 'ta_c mt05'})
            try_num -= 1
            if len(this_person_list) > 0:
                not_find = False
                for element in this_person_list:
                    try:
                        person_name = element.text
                        if person_name in all_person_set:
                            continue
                        search_index = search_360_index(person_name)
                        f.write(str(person_name)+'\t'+str(search_index)+'\n')
                        all_person_set.add(person_name)
                    except:
                        traceback.print_exc()
                        continue
                break
        if not_find:
            fail_url_dic[url] = 1
        print status, len(fail_url_dic), len(all_person_set)
        sleep(1)
    msgpack.dump(fail_url_dic, open('1905_fail_url.p', 'wb'))



def search_360_index(person_name):
    # 也是使用ajax的方法
    try:
        url = 'http://index.so.com/index.php?a=overviewJson' \
              '&q=%s&area=全国' % person_name
        req = urllib2.Request(url, None)
        response = urllib2.urlopen(req)
        html_doc = response.read()
        soup = BeautifulSoup(html_doc)
        dic = json.loads(soup.text)
        return dic.get('data')[0].get('data').get('month_index')
    except:
        return 0


if __name__ == '__main__':
    # crawler_baidu_person_list()
    # crawler_fans()
    # crawler_baidu_online()
    # print search_360_index(person_name='刘博')
    # pass
    crawler_1905_movie()
