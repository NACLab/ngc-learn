"""
Test the basic/key dynamics of an NGC Harmonium system constructed with Node(s)
and Cable(s), checked against an analytical/hand-coded RBM.
-------------------------------------------------------------------------------
This code runs the "harmonium test" for NGC projection and dynamics:
+ create a Harmonium from the Model Museum specifically set to be a mean-field
  model (no sampling in its latent feature detectors)
+ create a hand-coded RBM with weights set to be equal to those initialized in
  the Harmonium class (weight matrix, visible bias, hidden bias)
+ running both Harmonium class and hand-coded RBM, should get a Euclidean norm
  of the difference between the values returned by each for positive phase hidden
  state, negative phase  visible state, and negative phase hidden state, that
  is exactly equal to zero
+ running both Harmonium and hand-coded RBM, should get a Euclidean norm of the
  difference between the values returned by each for the visible bias gradient,
  the weight matrix W gradient, and the hidden bias gradient to be exactly zero
The test above is repeated for five iterations (or training steps). Since both
the Harmonium class and the RBM hand-coded object are updated with SGD, they
should NOT diverge and always yield Euclidean norms exactly equal to zero at
each iteration.
-------------------------------------------------------------------------------

The following models are tested:
a Harmonium (or 1-layer NGC generative model) built from the Model Museum

This (non-exhaustive) test script checks for qualitative irregularities in each
model's behavior and, indirectly, the functioning of ngc-learn's nodes and cables.
Please read the documentations in /docs/ for an overview and description/use of
Nodes and Cables as well as for practical use-cases through the
demonstrations (under the /examples directory).

As this is strictly a qualitative set of tests, it is up to the developer / user
to examine for specific irregularities.
"""

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
