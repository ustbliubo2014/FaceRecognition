# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: validate_on_lfw_new.py
@time: 2017/1/6 12:18
@contact: ustb_liubo@qq.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
from tensorflow.python.platform import gfile

def main(args):
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(None,args.image_size,args.image_size,3), name='input')
        with tf.Session() as sess:
            print('Loading graphdef: %s' % args.model_file)
            with gfile.FastGFile(os.path.expanduser(args.model_file),'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                return_elements = ['phase_train:0', 'embeddings:0']
                phase_train_placeholder, embeddings = tf.import_graph_def(graph_def, input_map={'input':images_placeholder}, 
                    return_elements=return_elements, name='import')
  
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

        # Get the paths for the corresponding images
        paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

        # Evaluate
        tpr, fpr, accuracy, val, val_std, far = lfw.validate(sess, paths, 
            actual_issame, args.seed, 60, 
            images_placeholder, phase_train_placeholder, embeddings, nrof_folds=args.lfw_nrof_folds)
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        
        facenet.plot_roc(fpr, tpr, 'NN4')
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model_file', type=str, 
        help='The graphdef for the model to be evaluated as a protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
