#-*- coding: utf-8 -*-
__author__ = 'liubo-it'

import urllib2
import base64
import sys
import traceback

def download(url, person_name, file_name):
    try:
        a = urllib2.urlopen(url, timeout=10)
        print person_name, file_name, url
        pic = base64.b64encode(a.read())
        print '\t'.join([person_name, file_name, pic])
    except:
        return


if __name__ == '__main__':
    for line in sys.stdin:
        try:
            tmp = line.rstrip().split()
            if len(tmp) >= 3:
                person_name = tmp[0]
                file_name = tmp[1]
                url = tmp[2]
                print person_name, file_name, url
                # download(url, person_name, file_name)
        except:
            continue

