# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: dmoz_spider.py
@time: 2016/8/2 13:48
@contact: ustb_liubo@qq.com
@annotation: dmoz_spider
"""
import sys
import logging
from logging.config import fileConfig
import os
import scrapy
from scrapy.spiders import Spider
from scrapy.selector import Selector
from tutorial.items import DmozItem

reload(sys)
sys.setdefaultencoding("utf-8")


class DmozSpider(Spider):
    name = "dmoz"
    allowed_domains = ["dmoz.org"]
    start_urls = [
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/",
    ]

    def parse(self, response):
        for sel in response.xpath('//ul/li'):
            item = DmozItem()
            item['title'] = sel.xpath('a/text()').extract()
            item['link'] = sel.xpath('a/@href').extract()
            item['desc'] = sel.xpath('text()').extract()
            yield item
            # guess-info-text