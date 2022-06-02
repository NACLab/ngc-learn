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
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np

from ngclearn.utils.config import Config
from ngclearn.density.gmm import GMM
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.stat_utils as stat
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_utils
from ngclearn.utils.data_utils import DataLoader

"""
################################################################################
Walkthrough #1 File:
Fitting a Gaussian mixture density/prior to a collected latent variable dataset,
i.e., latent vector codes extracted from a (pre-)trained NGC model.

Usage:
$ python fit_gmm.py --config=/path/to/analyze.cfg --gpu_id=0

@author Alexander Ororbia
################################################################################
"""

# GPU arguments
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["config=","gpu_id="])
# Collect arguments from argv
cfg_fname = None
use_gpu = False
gpu_id = -1
latents_fname =None
labels_fname =None
gmm_fname = None
for opt, arg in options:
    if opt in ("--config"):
        cfg_fname = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
        use_gpu = True

mid = gpu_id
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

args = None
if cfg_fname is not None:
    args = Config(cfg_fname)
    gmm_fname = args.getArg("gmm_fname") #"../models/pc_rao/mnist/gmm.pkl"
    latents_fname = args.getArg("latents_fname")
    labels_fname = args.getArg("labels_fname")

def sort_data_by_label(X, Y):
    lab_map = {}
    lab = tf.argmax(Y,axis=1)
    #print(lab.shape)
    for s in range(X.shape[0]):
        xs = tf.expand_dims(X[s,:],axis=0)
        ys = int(lab[s])
        blob = lab_map.get(ys)
        if blob is not None:
            blob = tf.concat([blob,xs],axis=0) #blob + xs
            lab_map[ys] = blob
        else:
            lab_map[ys] = xs
    return lab_map


#delimiter = "\t"
n_subset = 18000 #20000 #50000
with tf.device(gpu_tag):
    Y = None
    if labels_fname is not None:
        Y = np.load(labels_fname)
        Y = tf.cast(Y,dtype=tf.float32)
    z_lat = np.load(latents_fname)

    if Y is not None and n_subset < z_lat.shape[0]:
        print("original z_lat.shape = ",z_lat.shape)
        print("              Y.shape = ",Y.shape)
        print(" Label Reduction!")
        n_part = int(n_subset/Y.shape[1])
        print(" Subset.n_batch = ",n_part)
        data_map = sort_data_by_label(z_lat,Y)
        z_lat = None
        for c in range(Y.shape[1]):
            z_sub = data_map.get(c).numpy()
            ptrs = np.random.permutation(z_sub.shape[0])
            ptrs = ptrs[0:n_part]
            z_sub = tf.cast(z_sub[ptrs,:],dtype=tf.float32)
            if z_lat is not None:
                z_lat = tf.concat([z_lat,z_sub],axis=0)
            else:
                z_lat = z_sub
    else:
        if n_subset > 0:
            ptrs = np.random.permutation(z_lat.shape[0])[0:n_subset]
            z_lat = z_lat[ptrs,:]
        z_lat = tf.cast(z_lat,dtype=tf.float32)

    print("z_lat.shape = ",z_lat.shape)

    max_w = -10000.0
    min_w = 10000.0
    max_w = max(max_w, float(tf.reduce_max(z_lat)))
    min_w = min(min_w, float(tf.reduce_min(z_lat)))
    print("max_z = ", max_w)
    print("min_z = ", min_w)

    print(" > Estimating latent density...")
    n_comp = 75
    lat_density = GMM(k=n_comp)
    lat_density.fit(z_lat)

    print(" > Saving density estimator to: {0}".format("gmm.pkl"))
    fd = open("{0}".format(gmm_fname), 'wb')
    pickle.dump(lat_density, fd)
    fd.close()
