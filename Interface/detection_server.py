#coding : utf-8
__author__ = 'chenzhaocai'

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import time
import json
import base64
import caffe, os, sys, cv2
import tornado.ioloop
import tornado.web
import zlib
import urllib, urllib2
import msgpack

cfg.TEST.HAS_RPN = True
caffe.set_mode_gpu()
caffe.set_device(2)
cfg.GPU_ID = 2

class FaceDetector():
    def __init__(self):
        self.CLASSES = ('__background__', 'face')
        self.prototxt = os.path.join("/home/chenzhaocai/jupyter_notebook/faceDataSet/CNN_cascade/",
                                  "py-faster-rcnn-master/models/face/ZF/faster_rcnn_end2end/test.prototxt")
        self.caffemodel = os.path.join("/home/chenzhaocai/jupyter_notebook/faceDataSet/CNN_cascade/",
                                  "py-faster-rcnn-master/output/faster_rcnn_end2end/face_AFLW/",
                                  "zf_faster_rcnn_iter_26000.caffemodel")
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

    def detect(self, im):
        """Detect object classes in an image using pre-computed object proposals."""
        if im is None:
            return []
        # Detect all object classes and regress object bounds
        scores, boxes = im_detect(self.net, im)

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3

        cls_ind = 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        dets = dets[dets[:, -1] > CONF_THRESH] # x1, y1, x2, y2, score
        dets[:, 2] = dets[:, 2] - dets[:, 0]
        dets[:, 3] = dets[:, 3] - dets[:, 1]

        return dets[:, [0, 1, 2, 3]].tolist() # x1, y1, w, h

face_detector = FaceDetector()

received_images = []

class MainHandler(tornado.web.RequestHandler):
    def post(self):
        global received_images
        print "receive image", self.get_body_argument("image_id"), time.clock()

        try:
            image = base64.decodestring(zlib.decompress(base64.decodestring(self.get_body_argument("image"))))
            im = cv2.imdecode(np.fromstring(image, dtype=np.uint8), 1)
	    print im.shape
        except Exception, e:
	    print e
            im = None

        face_list = face_detector.detect(im)
        dt_rec_result = []
        dt_rec_result = face_list
        for face_rect in face_list:
            # x, y, w, h
            x, y, w, h = face_rect[:4]
            center_x = x + w/2
            center_y = y + h/2
            new_x_min = max(center_x - w * 0.65, 0)
            new_x_max = min(center_x + w * 0.65, im.shape[1])
            new_y_min = max(center_y - w * 0.65, 0)
            new_y_max = min(center_y + w * 0.65, im.shape[0])
            # face = im[face_rect[1]:face_rect[1]+face_rect[3], face_rect[0]:face_rect[0]+face_rect[2], :]
            face = im[new_y_min:new_y_max, new_x_min:new_x_max, :]
            # filter blurred image
            id = time.clock()
            b = self.blur(face)
            # print id, b
            if b < 500:
                continue
            # cv2.imwrite('/home/chenzhaocai/tmp/{0}_{1}.jpg'.format(id, b), face)
            # import random
            # cv2.imwrite("/home/chenzhaocai/tmp/{0}.jpg".format(random.randint(0, 100)), face)
            # recognize face
            # img_str = zlib.compress(cv2.imencode('.jpg', cv2.resize(face, (50, 50), interpolation=cv2.INTER_LINEAR))[1].tostring())
            img_str = zlib.compress(cv2.imencode('.jpg', face)[1].tostring())
            request = {
                "image_id": time.time(),
                "request_type": 'recognization',
                "image": base64.encodestring(img_str)
            }
            requestPOST = urllib2.Request(
                data=urllib.urlencode(request),
                url="http://10.160.164.25:6666/"
            )
            requestPOST.get_method = lambda : "POST"
            result = None
            try:
                result = urllib2.urlopen(requestPOST).read()
            except urllib2.HTTPError, e:
                print e.code
            except urllib2.URLError, e:
                print str(e)
            try:
                result = json.loads(result)["recognization"]
                person_name, score_proba, save_person_num, need_save = msgpack.loads(base64.b64decode(result))
            except:
                person_name, score_proba, save_person_num, need_save = '', '', '', ''
            dt_rec_result.append(face_rect + [person_name, score_proba, save_person_num, need_save])
        print dt_rec_result, time.clock()
        self.write(json.dumps({"detection": dt_rec_result}))

    def blur(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(image, cv2.CV_64F).var()
        return var

application = tornado.web.Application([
    (r"/", MainHandler),
])

if __name__ == "__main__":
    application.listen(9999)
    tornado.ioloop.IOLoop.instance().start()
