# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: mysql_test.py
@time: 2016/11/17 10:03
@contact: ustb_liubo@qq.com
@annotation: mysql_test
"""
import sys
import logging
from logging.config import fileConfig
import os
import pdb
from time import time
import MySQLdb

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


start = time()
# 打开数据库连接
db = MySQLdb.connect(
    host='10.16.66.44',
    port=3306,
    user='root',
    passwd='tianyan',
    db='face',
)

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# 使用execute方法执行SQL语句
tmp = cursor.execute("select img, name from person WHERE id in (select id from images WHERE is_moved = 1)")

info = cursor.fetchmany(tmp)
result = []

# SQL 更新语句
sql = "UPDATE images SET is_moved = 2 WHERE is_moved = 1"
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 提交到数据库执行
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()

# 关闭数据库连接
db.close()
end = time()

print 'all time :', (end - start)