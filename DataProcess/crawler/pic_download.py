# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/7/27 12:06
@contact: ustb_liubo@qq.com
@annotation: pic_download
"""
import urllib2
import base64
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def download(url, person_name, file_name):
    try:
        str_value = {'person_name': person_name, 'url': url,
                     'file_name': file_name}
        print 'person_name : %(person_name)s url : %(url)s file_name : ' \
              '$(file_name)s' % (str_value)
        a = urllib2.urlopen(url, timeout=10)
        pic = base64.b64encode(a.read())
        print '\t'.join([person_name, file_name, pic])
    except:
        return


if __name__ == '__main__':
    for line in sys.stdin:
        try:
            tmp = line.rstrip().split()
            if len(tmp) >= 3:
                person_name = str(tmp[0])
                file_name = str(tmp[1])
                url = str(tmp[2])
                download(url, person_name, file_name)
        except:
            continue
