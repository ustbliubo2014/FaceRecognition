# encoding: utf-8

"""
@author: liubo-it
@software: PyCharm Community Edition
@file: tmp.py
@time: 2016/8/23 18:32
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import logging
from logging.config import fileConfig
import os
from scipy.misc import imread
import cv2
import numpy as np
from matplotlib.pylab import *
reload(sys)
sys.setdefaultencoding("utf-8")
fileConfig('logger_config.ini')
logger_error = logging.getLogger('errorhandler')


# img = cv2.imread("20130627154241703.jpg", 0)
#
# x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
#
#
# absX = cv2.convertScaleAbs(x)   # 转回uint8
# absY = cv2.convertScaleAbs(y)
#
# dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#
# cv2.imshow("absX", absX)
# cv2.imshow("absY", absY)
#
# cv2.imshow("Result", dst)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


x = np.linspace(0, 10, 100)
y = 4 * np.sin(3*(4*np.sin(x)/np.pi)) / (3*np.pi)

figure()
plot(x, y, 'r')
xlabel('x')
ylabel('y')
title('title')
show()