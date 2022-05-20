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
args.setArg("samp_fx","identity")
rbm = Harmonium(args)

c = rbm.theta[0] + 0
W = rbm.theta[1] + 0
b = rbm.theta[2] + 0

opt = tf.keras.optimizers.SGD(0.1)

n_iter = 5 #100

x = np.zeros([1,5])
x[0,1] = 1.0
x[0,3] = 1.0
x = tf.cast(x,dtype=tf.float32)

goal = np.zeros([1,1])

for t in range(n_iter):
    h, xR, hR, dW, dc, db = calc(x, c, W, b)

    x_hat = rbm.settle(x)
    #print("{}: x_hat = {}".format(t, x_hat.numpy()))
    delta = rbm.calc_updates()

    # unit test - ngc-learn's theta should never drift apart from analytical RBM
    print("--------------- CHECK on Iteration {} ---------------".format(t))
    # print(tf.norm(c-rbm.theta[0]))
    # print(tf.norm(c-rbm.pos_phase.theta[0]))
    # print(tf.norm(c-rbm.neg_phase.theta[0]))
    print(" => Test for:  NGC.z1.phi(z) = RBM.hid-pos")
    check = tf.norm(rbm.pos_phase.extract("z1","phi(z)")-h)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.z0n.phi(z) = RBM.vis-neg")
    check = tf.norm(rbm.neg_phase.extract("z0n","phi(z)")-xR)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.z1n.phi(z) = RBM.his-neg")
    check = tf.norm(rbm.neg_phase.extract("z1n","phi(z)")-hR)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.vis_bias_grad - RBM.vis_bias_grad")
    check = tf.norm(-delta[0].numpy() - dc)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.W_grad - RBM.W_grad")
    check = tf.norm(-delta[1].numpy() - dW)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.hid_bias_grad - RBM.hid_bias_grad")
    check = tf.norm(-delta[2].numpy() - db)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    # print("h = ",tf.norm(rbm.pos_phase.extract("z1","phi(z)")-h))
    # print("xR = ",tf.norm(rbm.neg_phase.extract("z0n","phi(z)")-xR))
    # print("hR = ",tf.norm(rbm.neg_phase.extract("z1n","phi(z)")-hR))
    # print("dc = ",tf.norm(-delta[0].numpy() - dc))
    # print("dW = ",tf.norm(-delta[1].numpy() - dW))
    # print("db = ",tf.norm(-delta[2].numpy() - db))

    opt.apply_gradients(zip(delta, rbm.theta))
    rbm.neg_phase.set_theta(rbm.pos_phase.theta)
    rbm.clear()

    c = rbm.theta[0] + 0
    W = rbm.theta[1] + 0
    b = rbm.theta[2] + 0

#print("x = {}".format(x.numpy()))
