import os
import sys, getopt, optparse
import pickle
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import time

from sklearn.feature_extraction.image import extract_patches_2d

# import general simulation utilities
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader

# import harmonium from museum to train
from ngclearn.museum.harmonium import Harmonium

def calc(x, c, W, b):
    h = tf.nn.sigmoid(tf.matmul(x, W) + b)
    xR = tf.nn.sigmoid(tf.matmul(h, W, transpose_b=True) + c)
    hR = tf.nn.sigmoid(tf.matmul(xR, W) + b)
    dW = tf.matmul(x, h, transpose_a=True) - tf.matmul(xR, hR, transpose_a=True)
    dc = x - xR
    db = h - hR
    return h, xR, hR, dW, dc, db

args = Config()
args.setArg("batch_size",1)
args.setArg("z_dim",10)
args.setArg("x_dim",5)
args.setArg("seed",69)
args.setArg("wght_sd",0.02)
args.setArg("K",1)
args.setArg("act_fx","sigmoid")
args.setArg("out_fx","sigmoid")
rbm = Harmonium(args)

c = rbm.theta[0] + 0
W = rbm.theta[1] + 0
b = rbm.theta[2] + 0

opt = tf.keras.optimizers.SGD(0.1)

n_iter = 100 #500 #1000

x = np.zeros([1,5])
x[0,1] = 1.0
x[0,3] = 1.0
x = tf.cast(x,dtype=tf.float32)

for t in range(n_iter):
    x_hat = rbm.settle(x)
    print("{}: x_hat = {}".format(t, x_hat.numpy()))
    delta = rbm.calc_updates()

    #h, xR, hR, dW, dc, db = calc(x, c, W, b)

    opt.apply_gradients(zip(delta, rbm.theta))
    #rbm.pos_phase.apply_constraints()
    rbm.clear()

print("x = {}".format(x.numpy()))
