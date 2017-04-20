# encoding: utf-8

"""
@author: liubo
@software: PyCharm Community Edition
@file: AngleCalculateServer.py
@time: 2016/10/17 12:38
@contact: ustb_liubo@qq.com
@annotation: AngleCalculateServer
"""
import sys
sys.path.insert(0, '/home/liubo-it/FaceRecognization/')
import _init_paths
import time
import json
import base64
import tornado.ioloop
import tornado.web
import traceback
import msgpack_numpy
from ThreeD_Process.AngleCalculate import AngleCalculate
from ThreeD_Process.CriticalPointDetection import CriticalPointDetection

reload(sys)
sys.setdefaultencoding("utf-8")


check_port = 7788


class AngleCalculateServer(object):
    def __init__(self):
        self.criticalPointDetection = CriticalPointDetection()
        self.angle_calculate = AngleCalculate()
        self.angle_calculate.load_model()

    def calculate_angle(self, face_img, image_id):
        predict_points, pose_predict = self.angle_calculate.cal_angle(face_img)
        return pose_predict


angle_calculate_server = AngleCalculateServer()


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        request_type = self.get_body_argument('request_type')
        if request_type == 'check_pose':
            try:
                image_id = self.get_body_argument("image_id")
                face_img_str = self.get_body_argument("face_img_str")
                print "receive image", image_id, time.time()
                face_img = msgpack_numpy.loads(base64.b64decode(face_img_str))
                start = time.time()
                pose_predict = angle_calculate_server.calculate_angle(face_img, image_id)
                end = time.time()
                pose_predict = base64.b64encode(msgpack_numpy.dumps(pose_predict))
                print 'pose predict time :', (end - start)
                self.write(json.dumps({"pose_predict": pose_predict}))
            except:
                traceback.print_exc()
                return


if __name__ == '__main__':
    application = tornado.web.Application([(r"/", MainHandler), ])
    application.listen(check_port)
    tornado.ioloop.IOLoop.instance().start()
