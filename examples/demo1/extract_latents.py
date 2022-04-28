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
import time

from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.stat_utils as stat
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader
# import model from museum to train
from ngclearn.museum.gncn_t1 import GNCN_t1

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Demo/Tutorial #1 File:
Extracts/retrieves the latent representations of a (pre-)trained NGC model and
a provided data sample/pool, i.e., the MNIST database.

Usage:
$ python extract_latents.py --config=/path/to/analyze.cfg --gpu_id=0

@author Alexander Ororbia
################################################################################
"""

# read in configuration file and extract necessary simulation variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["config=","gpu_id="])
# GPU arguments
cfg_fname = None
use_gpu = False
gpu_id = -1
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
    #gpu_tag = '/GPU:0'
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

args = Config(cfg_fname)

out_dir = args.getArg("out_dir")
latents_fname = args.getArg("latents_fname")
model_fname = args.getArg("model_fname")
node_name = args.getArg("node_name") # z3
cmpt_name = args.getArg("cmpt_name") # phi(z)

batch_size = int(args.getArg("batch_size")) #128 #32
xfname = args.getArg("train_xfname")
print(" >> Loading data into memory...")
X = transform.binarize( tf.cast(np.load(xfname),dtype=tf.float32) ).numpy()
x_dim = X.shape[1]
train_set = DataLoader(design_matrices=[("z0",X)], batch_size=batch_size)

def extract_latents(agent, dataset, calc_ToD, verbose=False):
    """
        Extracts latent activities of an agent on a fixed-point data sample
    """
    latents = None
    ToD = 0.0
    Lx = 0.0
    N = 0.0
    for batch in dataset:
        x_name, x = batch[0]
        N += x.shape[0]
        x_hat = agent.settle(x) # conduct iterative inference
        lats = agent.ngc_model.extract(node_name, cmpt_name)
        if latents is not None:
            latents = tf.concat([latents,lats],axis=0)
        else:
            latents = lats
        ToD_t = calc_ToD(agent) # calc ToD
        # update tracked fixed-point losses
        Lx = tf.reduce_sum( metric.bce(x_hat, x) ) + Lx
        ToD = calc_ToD(agent) + ToD
        agent.clear()
        print("\r ToD {0}  Lx {1} over {2} samples...".format((ToD/(N * 1.0)), (Lx/(N * 1.0)), N),end="")
    print()
    Lx = Lx / N
    ToD = ToD / N
    return latents, ToD, Lx

################################################################################
# Start simulation
################################################################################
with tf.device(gpu_tag):

    def calc_ToD(agent):
        """Measures the total discrepancy (ToD) of a given NGC model"""
        ToD = 0.0
        L2 = agent.ngc_model.extract(node_name="e2", node_var_name="L")
        L1 = agent.ngc_model.extract(node_name="e1", node_var_name="L")
        L0 = agent.ngc_model.extract(node_name="e0", node_var_name="L")
        ToD = -(L0 + L1 + L2)
        return ToD

    agent = io_tools.deserialize(model_fname)

    sim_start_time = time.time()

    latents, ToD, Lx = extract_latents(agent, train_set, calc_ToD, verbose=True)

    sim_end_time = time.time()
    sim_time = sim_end_time - sim_start_time
    print("------------------------------------")
    sim_time_hr = (sim_time/3600.0) # convert time to hours
    print(" Trial.sim_time = {} h  ({} sec)".format(sim_time_hr, sim_time))

    #latents_fname = "{}{}_{}.npy".format(out_dir,node_name,trial)
    print(" >> Saving extracted latents to disk:  {0}".format(latents_fname))
    np.save(latents_fname, latents.numpy())
