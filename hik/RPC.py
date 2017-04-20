#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yaojian-xy <yaojian-xy@360.cn>

import random
import json
import traceback
from datetime import timedelta
import numpy as np

from flask import Flask, request, jsonify, make_response, request, current_app
from functools import update_wrapper
import time


# ref: documentation of Flask, [ http://flask.pocoo.org/docs/0.10/quickstart/ ]
app = Flask(__name__)
upload_path = "./data/"
# upload_path = "D:/vs/haicom/RPC/data/"

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator

def get_parameter(key, default=None, dtype=None):
    val = request.args.get(key)
    if val is None:
        val = default
    if dtype is not None:
        try:
            val = dtype(val)
        except:
            return default
    return val


def response(**msg):
    return jsonify(msg)




@app.route('/')
def index():
    return jsonify({'index': "index"})


@app.route('/upload', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', headers="Origin, X-Requested-With, Content-Type, Accept")
def render_action():
    """
        support params: (time, "time stamp in micro second")
                        (img, "image data")
    :return:
    """
    try:
        img_time = get_parameter("time")
        img_file = open("%s/%s.jpg" % (upload_path, img_time), 'wb')
        img_file.write(request.data)
        # 将文件名传给recognize模块,得到
        person1 = "10, 10, %d, %d, %s" % (90, 90, 'New Person A')
        person2 = "150, 150, %d, %d, %s" % (180, 180, 'New Person B')
        return ','.join(['0', person1, person2])
    except:
        print "upload error, detail=%s" % traceback.format_exc()
        return "-1"

      
if __name__ == '__main__':
    host, port = "10.138.107.156", 2220
    app.run(host=host, port=int(port), debug=False)
