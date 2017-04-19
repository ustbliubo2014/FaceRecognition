#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: process_files.py
@time: 2016/5/20 14:30
"""

import os

def process_file(file_name, person_name):

    lines = open(file_name).read().split('\n')
    f = open(file_name,'w')
    for line in lines:
        f.write(person_name+' '+line.rstrip()+'\n')
    f.close()


class Main(object):
    def __init__(self, files_folder):
        self.files_folder = files_folder

    def process_all_file(self):
        file_list = os.listdir(self.files_folder)
        for file_name in file_list:
            person_name = file_name[:-4]
            file_name = os.path.join(self.files_folder, file_name)
            process_file(file_name, person_name)
            print person_name


if __name__ == '__main__':
    files_folder = 'files'
    main = Main(files_folder)
    main.process_all_file()
