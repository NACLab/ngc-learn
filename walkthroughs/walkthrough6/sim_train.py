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

# import general simulation utilities
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader

# import model from museum to train
from ngclearn.museum.harmonium import Harmonium

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Walkthrough #6 File:
Trains/fits a harmonium (RBM) to a dataset of sensory patterns, e.g., the MNIST
database.

Usage:
$ python sim_train.py --config=/path/to/fit.cfg --gpu_id=0 --n_trials=1

@author Alexander Ororbia
################################################################################
"""

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
    #gpu_tag = '/GPU:0'
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

save_marker = 1

args = Config(cfg_fname)

model_type = args.getArg("model_type")
out_dir = args.getArg("out_dir")
batch_size = int(args.getArg("batch_size")) #128 #32
dev_batch_size = int(args.getArg("dev_batch_size")) #128 #32

eta = float(args.getArg("eta")) # learning rate/step size (optimzation)
num_iter = int(args.getArg("num_iter")) # num training iterations

# create training sample
xfname = args.getArg("train_xfname")
print(" >> Loading data into memory...")
X = transform.binarize( tf.cast(np.load(xfname),dtype=tf.float32) ).numpy()
x_dim = X.shape[1]
args.setArg("x_dim",x_dim)
train_set = DataLoader(design_matrices=[("z0",X)], batch_size=batch_size)

# create development/validation sample
xfname = args.getArg("dev_xfname") #"../data/mnist/validX.tsv"
X = transform.binarize( tf.cast(np.load(xfname),dtype=tf.float32) ).numpy()
dev_set = DataLoader(design_matrices=[("z0",X)], batch_size=dev_batch_size, disable_shuffle=True)

def eval_model(agent, dataset, verbose=False):
    """
        Evaluates performance of agent on this fixed-point data sample
    """
    Lx = 0.0 # metric/loss over entire data pool
    N = 0.0 # number samples seen so far
    for batch in dataset:
        x_name, x = batch[0]
        N += x.shape[0]
        x_hat = agent.settle(x, calc_update=False) # conduct iterative inference

        # update tracked fixed-point losses
        Lx = tf.reduce_sum( metric.bce(x_hat, x) ) + Lx
        agent.clear()
        if verbose == True:
            print("\r Lx {} over {} samples...".format((Lx/(N * 1.0)), N),end="")
    if verbose == True:
        print()
    Lx = Lx / N
    return Lx

################################################################################
# Start simulation
################################################################################
with tf.device(gpu_tag):

    for trial in range(n_trials): # for each trial
        agent = Harmonium(args)
        print(" >> Built Harmonium model w/ {} and {}".format(agent.pos_phase.name,
              agent.neg_phase.name)
              )

        eta_v  = tf.Variable( eta ) # set up optimization process
        opt = tf.keras.optimizers.Adam(eta_v)

        Lx_series = []
        vLx_series = []

        ############################################################################
        # create a  training loop
        #Lx = vLx = 0
        Lx = eval_model(agent, train_set, verbose=True)
        vLx = eval_model(agent, dev_set, verbose=True)
        print("{} | Lx = {} ; vLx = {}".format(-1, Lx, vLx))
        Lx_series.append(Lx)
        vLx_series.append(vLx)

        PATIENCE = 10 #5
        impatience = 0
        vLx_best = vLx
        sim_start_time = time.time()
        ########################################################################
        for i in range(num_iter): # for each training iteration/epoch
            Lx = 0.0
            n_s = 0
            # run single epoch/pass/iteration through dataset
            ####################################################################
            mark = 0
            for batch in train_set:
                n_s += batch[0][1].shape[0] # track num samples seen so far
                x_name, x = batch[0]
                mark += 1

                x_hat = agent.settle(x) # run RBM agent
                # update synaptic parameters given current model internal state
                delta = agent.calc_updates()
                opt.apply_gradients(zip(delta, agent.theta))
                agent.pos_phase.apply_constraints()
                agent.neg_phase.set_theta(agent.pos_phase.theta)
                agent.clear()

                Lx = tf.reduce_sum( metric.bce(x_hat, x) ) + Lx

                print("\r Lx {}  with {} samples seen...".format(
                      (Lx/(n_s * 1.0)), n_s), end=""
                      )
            ####################################################################
            print()
            Lx = Lx / (n_s * 1.0)
            # evaluate generalization ability on dev set
            vLx = eval_model(agent, dev_set)
            print("-------------------------------------------------")
            print("{} | Lx = {} ; vLx = {}".format(i, Lx, vLx))
            Lx_series.append(Lx)
            vLx_series.append(vLx)

            if i % save_marker == 0:
                np.save("{}Lx{}".format(out_dir, trial), np.array(Lx_series))
                np.save("{}vLx{}".format(out_dir, trial), np.array(vLx_series))

            if vLx < vLx_best:
                print(" -> Saving model checkpoint:  {} < {}".format(vLx, vLx_best))

                model_fname = "{}model{}.ngc".format(out_dir, trial)
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
