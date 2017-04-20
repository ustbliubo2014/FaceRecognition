# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: test_ratio.py
@time: 2016/11/11 11:40
@contact: ustb_liubo@qq.com
@annotation: test_acc : 验证跟踪的准确率
"""
import sys
from time import time
import cv2
from video_track.kcf import kcftracker
import pdb
import os

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
	global selectingObject, initTracking, onTracking, ix, iy, cx,cy, w, h

	if event == cv2.EVENT_LBUTTONDOWN:
		selectingObject = True
		onTracking = False
		ix, iy = x, y
		cx, cy = x, y

	elif event == cv2.EVENT_MOUSEMOVE:
		cx, cy = x, y

	elif event == cv2.EVENT_LBUTTONUP:
		selectingObject = False
		if(abs(x-ix)>10 and abs(y-iy)>10):
			w, h = abs(x - ix), abs(y - iy)
			ix, iy = min(x, ix), min(y, iy)
			initTracking = True
		else:
			onTracking = False

	elif event == cv2.EVENT_RBUTTONDOWN:
		onTracking = False
		if(w>0):
			ix, iy = x-w/2, y-h/2
			initTracking = True


if __name__ == '__main__':
    tracker = kcftracker.KCFTracker(False, True, False)
    folder = 'C:\Users\liubo\Desktop\picture/2016-11-11-10-40/'
    first_pic = '%s/1478832096.2.jpg'%folder
    frame = cv2.imread(first_pic)
    tracker.init([0, 0, frame.shape[0], frame.shape[1]], frame)

    pic_list = os.listdir(folder)
    pic_list.sort()
    for pic in pic_list:
        frame = cv2.imread(os.path.join(folder, pic))
        tracker.update(frame)

