# -*coding:utf-8*-
__author__ = 'liubo-it'

nb_filters = 64
nb_conv = 3
nb_pool = 2
dropout = 0.5
hidden_num = 1024
lr = 0.01
dim_ordering = 'th'
# 'th' (channels, width, height) or 'tf' (width, height, channels)
WEIGHT_DECAY = 0.    # L2 regularization factor
USE_BN = True        # whether to use batch normalization