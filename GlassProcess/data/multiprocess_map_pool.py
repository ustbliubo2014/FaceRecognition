# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: multipr.py
@time: 2016/8/23 16:55
@contact: ustb_liubo@qq.com
@annotation: multiprocess_map_pool : 利用map实现多进程并行
"""
import sys
import logging
from logging.config import fileConfig
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from time import time
import urllib2
import traceback

reload(sys)
sys.setdefaultencoding("utf-8")


def download(args):
    try:
        pic_url, name = args
        a = urllib2.urlopen(pic_url, timeout=10)
        f = open(name, 'w')
        f.write(a.read())
        f.close()
    except:
        traceback.print_exc()
        return


def crawler_img(person_name='女黑框眼镜', limit=3000, result_file='woman_glass.txt'):
    f = open(result_file, 'w')
    pic_url_list = set()
    try:
        for page_index in range(100):
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
                    if start_index != -1:   # 还有objURL
                        end_index = str_soup.find(',', start_index)
                        if end_index != -1:
                            pic_url = str_soup[start_index:end_index][9:-1]
                            pic_url_list.add(pic_url)
                            last_index = end_index
                            f.write(pic_url+'\n')
                    else:
                        break
            except:
                continue
            if len(pic_url_list) > limit:
                break
    except:
        return pic_url_list
    f.close()
    return pic_url_list


if __name__ == '__main__':
    process_num = 20
    pool = Pool(process_num)
    start = time()
    url_file = 'woman_glass.txt'
    pic_folder = 'woman_glass'
    url_list = []
    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)
    index = 0
    for line in open('woman_glass.txt'):
        name = os.path.join(pic_folder, str(index)+'.jpg')
        index += 1
        url_list.append((line, name))
    pool.map(download, url_list)
    pool.close()
    pool.join()   # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    end = time()
    print 'process_num', process_num, 'use all time',(end-start)

