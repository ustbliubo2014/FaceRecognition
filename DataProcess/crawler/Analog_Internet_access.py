# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: Analog_Internet_access.py
@time: 2016/8/1 11:04
@contact: ustb_liubo@qq.com
@annotation: Analog_Internet_access
"""
import random
import socket
import urllib2
import cookielib
import pdb


class BrowserBase(object):

    def __init__(self):
        socket.setdefaulttimeout(20)

    def speak(self,name,content):
        print '[%s]%s' %(name,content)

    def openurl(self,url):
        """
        打开网页
        """
        cookie_support= urllib2.HTTPCookieProcessor(cookielib.CookieJar())
        self.opener = urllib2.build_opener(cookie_support, urllib2.HTTPHandler)
        urllib2.install_opener(self.opener)
        user_agents = [
                    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
                    'Opera/9.25 (Windows NT 5.1; U; en)',
                    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
                    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
                    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
                    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
                    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
                    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 ",

                    ]

        agent = random.choice(user_agents)
        self.opener.addheaders = [("User-agent",agent),
                                  ("Accept","*/*"),
                                  ('Referer', 'http://www.google.com')
        ]
        try:
            res = self.opener.open(url)
            return res.read()
        except:
            return None
        # except Exception,e:
        #     self.speak(str(e)+url)
        #     raise Exception



if __name__=='__main__':
    baidu_url = 'http://image.baidu.com/n/pc_search?queryImageUrl=http%3A%2F%2Fphotocdn.sohu.com%2F20141029%2FImg405580701.jpg&querySign=&simid=&fm=index&pos=&uptype=paste'

    splider=BrowserBase()
    # splider.openurl('http://blog.csdn.net/v_JULY_v/archive/2010/11/27/6039896.aspx')
    tmp = splider.openurl(baidu_url)
