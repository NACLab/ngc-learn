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
from ngclearn.museum.gncn_t1_ffm import GNCN_t1_FFM

seed = 69
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Demo/Tutorial #3 File:
Fits an NGC classifier to the MNIST database.

Usage:
$ python sim_train.py --config=gncn_t1_ffm/fit.cfg --gpu_id=0

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

out_dir = args.getArg("out_dir")
batch_size = int(args.getArg("batch_size")) #128 #32
dev_batch_size = int(args.getArg("dev_batch_size")) #128 #32

eta = float(args.getArg("eta")) # learning rate/step size (optimzation)
num_iter = int(args.getArg("num_iter")) # num training iterations

# create training sample
xfname = args.getArg("train_xfname")
yfname = args.getArg("train_yfname")
print(" >> Loading data into memory...")
X = ( tf.cast(np.load(xfname),dtype=tf.float32) ).numpy()
x_dim = X.shape[1]
args.setArg("x_dim",x_dim)
Y = ( tf.cast(np.load(yfname),dtype=tf.float32) ).numpy()
y_dim = Y.shape[1]
args.setArg("y_dim",y_dim)
train_set = DataLoader(design_matrices=[("z3",X),("z0",Y)], batch_size=batch_size)

# create development/validation sample
xfname = args.getArg("dev_xfname")
yfname = args.getArg("dev_yfname")
X = ( tf.cast(np.load(xfname),dtype=tf.float32) ).numpy()
Y = ( tf.cast(np.load(yfname),dtype=tf.float32) ).numpy()
dev_set = DataLoader(design_matrices=[("z3",X),("z0",Y)], batch_size=dev_batch_size, disable_shuffle=True)

def eval_model(agent, dataset, calc_ToD, verbose=False):
    """
        Evaluates performance of agent on this fixed-point data sample
    """
    ToD = 0.0 # total disrepancy over entire data pool
    Ly = 0.0 # metric/loss over entire data pool
    Acc = 0.0
    N = 0.0 # number samples seen so far
    for batch in dataset:
        x_name, x = batch[0]
        y_name, y = batch[1]
        N += x.shape[0]

        # run (fast) classifier or p(y|x)
        y_hat = agent.predict(x)

        # update tracked fixed-point losses
        Ly = tf.reduce_sum( metric.cat_nll(y_hat, y) ) + Ly

        # track raw accuracy
        y_ind = tf.cast(tf.argmax(y,1),dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(y_hat,1),dtype=tf.int32)
        comp = tf.cast(tf.equal(y_pred,y_ind),dtype=tf.float32)
        Acc += tf.reduce_sum( comp )

        agent.clear()
        if verbose == True:
            print("\r Acc {}  Ly {} over {} samples...".format((Acc/(N * 1.0)), (Ly/(N * 1.0)), N),end="")
    if verbose == True:
        print()
    Ly = Ly / N
    Acc = Acc / N
    return ToD, Ly, Acc

################################################################################
# Start simulation
################################################################################
with tf.device(gpu_tag):
    def calc_ToD(agent):
        """Measures the total discrepancy (ToD) of a given NGC model"""
        ToD = 0.0
        #print("TOD:")
        L2 = agent.ngc_model.extract(node_name="e2", node_var_name="L")
        #tf.print(L2)
        L1 = agent.ngc_model.extract(node_name="e1", node_var_name="L")
        #tf.print(L1)
        L0 = agent.ngc_model.extract(node_name="e0", node_var_name="L")
        #tf.print(L0)
        ToD = -(L0 + L1 + L2)
        return float(ToD)

    for trial in range(n_trials): # for each trial
        agent = GNCN_t1_FFM(args) # set up NGC model

        eta_v  = tf.Variable( eta ) # set up optimization process
        opt = tf.keras.optimizers.Adam(eta_v)#, beta_1=0.9, beta_2=0.999, epsilon=1e-6)
        print(" >> Built model = {}".format(agent.ngc_model.name))

        Ly_series = []
        Acc_series = []
        vLy_series = []
        vAcc_series = []

        ############################################################################
        # create a  training loop
        ToD, Ly, Acc = eval_model(agent, train_set, calc_ToD, verbose=True)
        vToD, vLy, vAcc = eval_model(agent, dev_set, calc_ToD, verbose=True)
        print("{} | Acc = {}  Ly = {} ; vAcc = {}  vLy = {}".format(-1, Acc, Ly, vAcc, vLy))

        Ly_series.append(Ly)
        Acc_series.append(Acc)
        vLy_series.append(vLy)
        vAcc_series.append(Acc)

        PATIENCE = 30 #20
        impatience = 0
        vAcc_best = vAcc
        sim_start_time = time.time()
        ########################################################################
        for i in range(num_iter): # for each training iteration/epoch
            ToD = 0.0
            Ly = 0.0
            n_s = 0
            # Run single epoch/pass/iteration through dataset
            ####################################################################
            mark = 0.0
            inf_time = 0.0
            for batch in train_set:
                n_s += batch[0][1].shape[0] # track num samples seen so far
                x_name, x = batch[0]
                y_name, y = batch[1]
                mark += 1

                # run ancestral projection to get initial conditions
                y_hat_ = agent.predict(x) # run p(y|x)
                mu2 = agent.ngc_sampler.extract("s2","z") # extract value of s2
                mu1 = agent.ngc_sampler.extract("s1","z") # extract value of s1
                mu0 = agent.ngc_sampler.extract("s0","z") # extract value of s0

                agent.ngc_model.inject([("mu2", "z", mu2), ("z2", "z", mu2),
                                        ("mu1", "z", mu1), ("z1", "z", mu1),
                                        ("mu0", "z", mu0)])

                # conduct iterative inference
                inf_t = time.time()
                y_hat = agent.settle(x, y)
                inf_t = time.time() - inf_t
                inf_time += inf_t

                ToD_t = calc_ToD(agent) # calc ToD
                Ly = tf.reduce_sum( metric.cat_nll(y_hat, y) ) + Ly

                # update synaptic parameters given current model internal state
                delta = agent.calc_updates(avg_update=False)
                opt.apply_gradients(zip(delta, agent.ngc_model.theta))
                agent.ngc_model.apply_constraints()
                agent.clear()

                ToD = ToD_t + ToD
                print("\r train.ToD {}  Ly {}  with {} samples seen (time = {} s)".format(
                      (ToD/(n_s * 1.0)), (Ly/(n_s * 1.0)), n_s, (inf_time/mark)),
                      end=""
                      )
            ####################################################################
            print()
            ToD = ToD / (n_s * 1.0)
            Ly = Ly / (n_s * 1.0)

            ToD, Ly, Acc = eval_model(agent, train_set, calc_ToD, verbose=True)
            # evaluate generalization ability on dev set
            vToD, vLy, vAcc = eval_model(agent, dev_set, calc_ToD, verbose=True)
            print("-------------------------------------------------")
            print("{} | Acc = {}  Ly = {} ; vAcc = {}  vLy = {}".format(i, Acc, Ly, vAcc, vLy))

            Ly_series.append(Ly)
            Acc_series.append(Acc)
            vLy_series.append(vLy)
            vAcc_series.append(vAcc)

            if i % save_marker == 0:
                np.save("{}Ly{}".format(out_dir, trial), np.array(Ly_series))
                np.save("{}Acc{}".format(out_dir, trial), np.array(Acc_series))
                np.save("{}vLy{}".format(out_dir, trial), np.array(vLy_series))
                np.save("{}vAcc{}".format(out_dir, trial), np.array(vAcc_series))

            if vAcc >= vAcc_best:
                print(" -> Saving model checkpoint:  {} >= {}".format(vAcc, vAcc_best))

                model_fname = "{}model{}.ngc".format(out_dir, trial)
                io_tools.serialize(model_fname, agent)

                vAcc_best = vAcc
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
        print(" Trial.sim_time = {} h  ({} sec)  Best Acc = {}".format(sim_time_hr, sim_time, vAcc_best))
