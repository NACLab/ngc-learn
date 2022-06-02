"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""

import os
import sys, getopt, optparse
import pickle
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#cmap = plt.cm.jet

import ngclearn.utils.io_utils as io_utils
import ngclearn.utils.transform_utils as transform

"""
################################################################################
Walkthrough #6 File:
Visualizes the filters of a trained harmonium (RBM) model.
Loads in NGC from filename "model_fname" and saves a set of samples from the
model's underlying block Gibbs sampler.

Usage:
$ python sample_rbm.py --model_fname=/path/to/modelN.ngc --output_dir=/path/to/output_dir/

@author Alexander Ororbia
################################################################################
"""

### specify output directory and model filename ###
viz_encoder = False
output_dir = "rbm/"
model_fname = "rbm/model0.ngc"
# read in configuration file and extract necessary simulation variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["model_fname=","output_dir="])
for opt, arg in options:
    if opt in ("--model_fname"):
        model_fname = arg.strip()
    elif opt in ("--output_dir"):
        output_dir = arg.strip()

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

### load in NGC model ###
model = io_utils.deserialize(model_fname)
x_dim = model.pos_phase.getNode("z0").dim
px = py = int(np.sqrt(x_dim)) #16
print("Outputs will be of shape: {}x{} ".format(px,py))

xfname = "../data/mnist/trainX.npy"
yfname = "../data/mnist/trainY.npy"
print(" >> Loading data into memory...")
X = transform.binarize( tf.cast(np.load(xfname),dtype=tf.float32) ).numpy()
Y = np.load(yfname)

K = 80 # how many steps to run the block Gibbs sampler
seed = 3333 # 9431
model.seed = seed
model.pos_phase.compile(batch_size=1, use_graph_optim=False)
model.neg_phase.compile(batch_size=1, use_graph_optim=False)

tf.random.set_seed(seed=seed)
np.random.seed(seed)

with tf.device('/CPU:0'):
    #datamap = sort_data_by_label(X,Y)
    ptrs = np.random.permutation(X.shape[0])[0:3]
    print(ptrs)
    for s in range(len(ptrs)):
        ptr = int(ptrs[s])
        x_init = np.expand_dims(X[ptr,:],axis=0)

        fname = "{}data{}.png".format(output_dir, s)
        io_utils.plot_sample_img(x_init, px, py, fname, plt)

        samples = None
        sample_chain = model.sample(K, x_sample=x_init)
        for t in range(len(sample_chain)):
            if t % 20 == 0:
                x_t = sample_chain[t].numpy()
                if samples is not None:
                    samples = tf.concat([samples,x_t],axis=0)
                else:
                    samples = x_t
                # fname = "{}sample{}.png".format(output_dir, t)
                # print("=> Saving sample:  ",fname)
                # io_utils.plot_sample_img(x_t, px, py, fname, plt)
        fname = "{}samples{}.png".format(output_dir, s)
        io_utils.plot_img_grid(samples.numpy(), fname, nx=1, ny=4, px=px, py=py, plt=plt)
