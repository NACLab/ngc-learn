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
from ngclearn.museum.gncn_t1 import GNCN_t1
from ngclearn.museum.gncn_t1_sigma import GNCN_t1_Sigma
from ngclearn.museum.gncn_pdh import GNCN_PDH

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Demo/Tutorial #1 File:
Trains/fits an NGC model to a dataset of sensory patterns, e.g., the MNIST
database. Note that this script will sequentially run multiple trials/seeds if an
experimental multi-trial setup is required (the tutorial only requires 1 trial).

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

def eval_model(agent, dataset, calc_ToD, verbose=False):
    """
        Evaluates performance of agent on this fixed-point data sample
    """
    ToD = 0.0 # total disrepancy over entire data pool
    Lx = 0.0 # metric/loss over entire data pool
    N = 0.0 # number samples seen so far
    for batch in dataset:
        x_name, x = batch[0]
        N += x.shape[0]
        x_hat = agent.settle(x) # conduct iterative inference
        # update tracked fixed-point losses
        Lx = tf.reduce_sum( metric.bce(x_hat, x) ) + Lx
        ToD = calc_ToD(agent) + ToD  # calc ToD
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
    def calc_ToD(agent):
        """Measures the total discrepancy (ToD) of a given NGC model"""
        ToD = 0.0
        L2 = agent.ngc_model.extract(node_name="e2", node_var_name="L")
        L1 = agent.ngc_model.extract(node_name="e1", node_var_name="L")
        L0 = agent.ngc_model.extract(node_name="e0", node_var_name="L")
        ToD = -(L0 + L1 + L2)
        return ToD

    for trial in range(n_trials): # for each trial
        agent = None # set up NGC model
        print(" >> Building ",model_type)
        if model_type == "GNCN_t1":
            agent = GNCN_t1(args)
        elif model_type == "GNCN_t1_Sigma":
            agent = GNCN_t1_Sigma(args)
        elif model_type == "GNCN_PDH" or model_type == "GNCN_t2_LSigma_PDH":
            agent = GNCN_PDH(args)

        eta_v  = tf.Variable( eta ) # set up optimization process
        #opt = tf.compat.v1.train.AdamOptimizer(learning_rate=eta_v,beta1=0.9, beta2=0.999, epsilon=1e-6)
        opt = tf.keras.optimizers.Adam(eta_v)

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

        PATIENCE = 10 #5
        impatience = 0
        vLx_best = vLx
        sim_start_time = time.time()
        ########################################################################
        for i in range(num_iter): # for each training iteration/epoch
            ToD = 0.0
            Lx = 0.0
            n_s = 0
            # run single epoch/pass/iteration through dataset
            ####################################################################
            for batch in train_set:
                n_s += batch[0][1].shape[0] # track num samples seen so far
                x_name, x = batch[0]
                x_hat = agent.settle(x) # conduct iterative inference
                ToD_t = calc_ToD(agent) # calc ToD
                Lx = tf.reduce_sum( metric.bce(x_hat, x) ) + Lx
                # update synaptic parameters given current model internal state
                delta = agent.calc_updates()
                opt.apply_gradients(zip(delta, agent.ngc_model.theta))
                agent.ngc_model.apply_constraints()
                agent.clear()

                ToD = ToD_t + ToD
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
