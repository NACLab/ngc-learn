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

from sklearn.feature_extraction.image import extract_patches_2d

# import general simulation utilities
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader

# import sparse coding model from museum to train
from ngclearn.museum.gncn_t1_sc import GNCN_t1_SC

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Demo/Tutorial #4 File:
Trains/fits an sparse coding model to the "natural scenes" dataset.
Note that this script will sequentially run multiple trials/seeds if an
experimental multi-trial setup is required (the tutorial only requires 1 trial).

Usage:
$ python sim_train.py --config=/path/to/fit.cfg --gpu_id=0 --n_trials=1

@author Alexander Ororbia
################################################################################
"""

def generate_patch_set(x_batch, patch_size, max_patches=50):
    """
    Uses scikit-learn's patch creation function to generate a set of (px x py) patches.
    Note: this routine also subtracts each patch's mean from itself.
    """
    px = py = int(np.sqrt(x_batch.shape[1])) # get image shape of the data
    p_batch = None
    for s in range(x_batch.shape[0]):
        xs = x_batch[s,:]
        xs = xs.reshape(px, py)
        patches = extract_patches_2d(xs, patch_size, max_patches=max_patches)#, random_state=69)
        patches = np.reshape(patches, (len(patches), -1)) # flatten each patch in set
        if s > 0:
            p_batch = np.concatenate((p_batch,patches),axis=0)
        else:
            p_batch = patches
    mu = np.mean(p_batch,axis=1,keepdims=True)
    p_batch = p_batch - mu
    return p_batch

# read in configuration file and extract necessary simulation variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["config=","gpu_id=","n_trials="])
# GPU arguments
cfg_fname = None
use_gpu = False
n_trials = 1
gpu_id = -1
for opt, arg in options:
    if opt in ("--config"):
        cfg_fname = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
        use_gpu = True
    elif opt in ("--n_trials"):
        n_trials = int(arg.strip())
mid = gpu_id
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

save_marker = 1

args = Config(cfg_fname)

model_type = args.getArg("model_type")
out_dir = args.getArg("out_dir")
batch_size = 1 # we want to randomly sample 1 image at a time
dev_batch_size = int(args.getArg("dev_batch_size"))

opt_type = args.getArg("opt_type")
eta = float(args.getArg("eta")) # learning rate/step size (optimzation)
num_iter = int(args.getArg("num_iter")) # num training iterations
lmda = 0.0
if args.hasArg("lambda"):
    lmda = float(args.getArg("lambda"))
else:
    lmda = float(args.getArg("thr_lambda"))

# create training sample
xfname = args.getArg("train_xfname")
print(" >> Loading data into memory...")
X = tf.cast(np.load(xfname),dtype=tf.float32).numpy()
x_dim = X.shape[1]
print(X.shape)
train_set = DataLoader(design_matrices=[("z0",X)], batch_size=batch_size)

# create development/validation sample
xfname = args.getArg("dev_xfname")
X = tf.cast(np.load(xfname),dtype=tf.float32).numpy()
dev_set = DataLoader(design_matrices=[("z0",X)], batch_size=dev_batch_size, disable_shuffle=True)

patch_size = (16,16) # 16x16 patches - this follows the setup of (Olshausen &amp; Field, 1996)
num_patches = 250

args.setArg("x_dim",(patch_size[0] * patch_size[1]))

def eval_model(agent, dataset, calc_ToD, verbose=False):
    """
        Evaluates performance of agent on this fixed-point data sample
    """
    ToD = 0.0 # total disrepancy over entire data pool
    Lx = 0.0 # metric/loss over entire data pool
    N = 0.0 # number samples seen so far
    for batch in dataset:
        x_name, x = batch[0]
        # generate patches on-the-fly for sample x
        x_p = generate_patch_set(x, patch_size, 50)
        x = x_p
        N += x.shape[0]

        x_hat = agent.settle(x) # conduct iterative inference
        # update tracked fixed-point losses
        Lx = tf.reduce_sum( metric.mse(x_hat, x) ) + Lx
        ToD = calc_ToD(agent, lmda) + ToD  # calc ToD
        agent.clear()
        if verbose == True:
            print("\r ToD {0}  Lx {1} over {2} samples...".format((ToD/(N * 1.0)), (Lx/(N * 1.0)), N),end="")
    if verbose == True:
        print()
    Lx = Lx / N
    ToD = ToD / N
    return ToD, Lx

################################################################################
# Start simulation
################################################################################
with tf.device(gpu_tag):
    def calc_ToD(agent, lmda):
        """Measures the total discrepancy (ToD), or negative energy, of an NGC system"""
        z1 = agent.ngc_model.extract(node_name="z1", node_var_name="z")
        e0 = agent.ngc_model.extract(node_name="e0", node_var_name="phi(z)")
        z1_sparsity = tf.reduce_sum(tf.math.abs(z1)) * lmda # sparsity penalty term
        L0 = tf.reduce_sum(tf.math.square(e0)) # reconstruction term
        ToD = -(L0 + z1_sparsity)
        return ToD

    for trial in range(n_trials): # for each trial
        print(" >> Building ",model_type) # set up NGC model
        agent = GNCN_t1_SC(args)
        print(" >> Built model = {}".format(agent.ngc_model.name))

        eta_v  = tf.Variable( eta ) # set up optimization process
        #opt = tf.compat.v1.train.AdamOptimizer(learning_rate=eta_v,beta1=0.9, beta2=0.999, epsilon=1e-6)
        if opt_type == "adam":
            opt = tf.keras.optimizers.Adam(eta_v)
        else: # default is SGD
            opt = tf.keras.optimizers.SGD(eta_v)

        Lx_series = []
        ToD_series = []
        vLx_series = []
        vToD_series = []

        ############################################################################
        # create a  training loop
        ToD, Lx = eval_model(agent, train_set, calc_ToD, verbose=True)
        vToD, vLx = eval_model(agent, dev_set, calc_ToD, verbose=True)
        print("{} | ToD = {}  Lx = {} ; vToD = {}  vLx = {}".format(-1, ToD, Lx, vToD, vLx))
        Lx_series.append(Lx)
        ToD_series.append(ToD)
        vLx_series.append(vLx)
        vToD_series.append(vToD)

        PATIENCE = 1000 #5
        impatience = 0
        vLx_best = vLx
        sim_start_time = time.time()
        ########################################################################
        for i in range(num_iter): # for each training iteration/epoch
            ToD = 0.0 # track window average of online ToD
            Lx = 0.0 # track window average of online mean patch loss
            n_s = 0
            # run single epoch/pass/iteration through dataset
            ####################################################################
            for batch in train_set:
                x_name, x = batch[0]
                # generate patches on-the-fly for sample x
                x_p = generate_patch_set(x, patch_size, num_patches)
                x = x_p
                n_s += x.shape[0] # track num samples seen so far

                x_hat = agent.settle(x) # conduct iterative inference
                ToD_t = calc_ToD(agent, lmda) # calc ToD

                ToD = ToD_t + ToD
                Lx = tf.reduce_sum( metric.mse(x_hat, x) ) + Lx

                # update synaptic parameters given current model internal state
                delta = agent.calc_updates(avg_update=False)
                opt.apply_gradients(zip(delta, agent.ngc_model.theta))
                agent.ngc_model.apply_constraints()
                agent.clear()

                print("\r train.ToD {0}  Lx {1}  with {2} samples seen...".format(
                      (ToD/(n_s * 1.0)), (Lx/(n_s * 1.0)), n_s),
                      end=""
                      )
            ####################################################################
            print()
            ToD = ToD / (n_s * 1.0)
            Lx = Lx / (n_s * 1.0)
            # evaluate generalization ability on dev set
            vToD, vLx = eval_model(agent, dev_set, calc_ToD)
            print("-------------------------------------------------")
            print("{} | ToD = {}  Lx = {} ; vToD = {}  vLx = {}".format(
                  i, ToD, Lx, vToD, vLx)
                  )
            Lx_series.append(Lx)
            ToD_series.append(ToD)
            vLx_series.append(vLx)
            vToD_series.append(vToD)

            if i % save_marker == 0:
                np.save("{}Lx{}".format(out_dir, trial), np.array(Lx_series))
                np.save("{}ToD{}".format(out_dir, trial), np.array(ToD_series))
                np.save("{}vLx{}".format(out_dir, trial), np.array(vLx_series))
                np.save("{}vToD{}".format(out_dir, trial), np.array(vToD_series))

            model_fname = "{}model{}.ngc".format(out_dir, trial)
            io_tools.serialize(model_fname, agent)

            if vLx < vLx_best:
                print(" -> Saving model checkpoint:  {} < {}".format(vLx, vLx_best))

                model_fname = "{}model_best{}.ngc".format(out_dir, trial)
                io_tools.serialize(model_fname, agent)

                vLx_best = vLx
                impatience = 0
            else: # execute early-stopping (through a patience mechanism)
                impatience += 1
                if impatience >= PATIENCE: # patience exceeded, so early stop
                    print(" > Executed early stopping!!")
                    break
        ########################################################################
        sim_end_time = time.time()
        sim_time = sim_end_time - sim_start_time
        print("------------------------------------")
        sim_time_hr = (sim_time/3600.0) # convert time to hours
        print(" Trial.sim_time = {} h  ({} sec)".format(sim_time_hr, sim_time))
