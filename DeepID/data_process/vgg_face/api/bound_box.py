#!/usr/bin/env python
# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: bound_box.py
@time: 2016/5/19 17:45
"""

from scipy.misc import imread, imsave
def bound_image(left, top, right, bottom, img, new_img):
    arr = imread(img)
    print arr.shape
    bound_arr = arr[left:right, top:bottom, :]
    print bound_arr.shape
    imsave(new_img, bound_arr)

class Main(object):
    def __init__(self):
        pass


if __name__ == '__main__':


    # # [5, 162, 325, 325]
    # left = 162
    # top = 5
    # right = 325+5
    # bottom = 325+162
    # img = '00000049.png'
    # new_img = '00000049_bound.png'
    # bound_image(left, top, right, bottom, img, new_img)
    #
    # # 51, 180, 232, 232
    # left = 180
    # top = 51
    # right = 232+180
    # bottom = 232+51
    # img = '00000066.png'
    # new_img = '00000066_bound.png'
    # bound_image(left, top, right, bottom, img, new_img)
    #
    # # [138, 53, 92, 99]
    #
    # left = 53
    # top = 138
    # right = 53+99
    # bottom = 138+92
    # img = '00000068.png'
    # new_img = '00000068_bound.png'
    # bound_image(left, top, right, bottom, img, new_img)

    # [184, 93, 165, 165]
    # left = 93
    # top = 184
    # right = 93+165
    # bottom = 184+165
    # img = '00000040.png'
    # new_img = '00000040_bound.png'
    # bound_image(left, top, right, bottom, img, new_img)

    # [386, 69, 82, 82]
    left = 69
    top = 386
    right = 69+82
    bottom = 386+82
    img = '00000620.png'
    new_img = '00000620_bound.png'
    bound_image(left, top, right, bottom, img, new_img)
# Alexander_Gould
    # [145, 105, 59, 59]
    left = 105
    top = 145
    right = 105+59
    bottom = 145+59
    img = '00000620.png'
    new_img = '00000620_bound_1.png'
    bound_image(left, top, right, bottom, img, new_img)

    # 129.61 95.32 205.78 171.48
    left = 95
    top = 129
    right = 171
    bottom = 205
    img = '00000620.png'
    new_img = '00000620_bound_2.png'
    bound_image(left, top, right, bottom, img, new_img)