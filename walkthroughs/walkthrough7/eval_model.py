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
import ngclearn.utils.viz_utils as viz
from ngclearn.utils.data_utils import DataLoader

seed = 69
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Walkthrough #7 File:
Evaluates a trained SNN classifier on the MNIST database test-set.

Usage:
$ python eval_train.py --config=/path/to/file.cfg --gpu_id=0

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

save_marker = 1

args = Config(cfg_fname)

out_dir = args.getArg("out_dir")
model_fname = args.getArg("model_fname")
dev_batch_size = int(args.getArg("dev_batch_size")) #128 #32

# create development/validation sample
xfname = args.getArg("test_xfname")
yfname = args.getArg("test_yfname")
print(" Evaluating model on X.fname = {}".format(xfname))
print("                     Y.fname = {}".format(yfname))
X = ( tf.cast(np.load(xfname),dtype=tf.float32) ).numpy()
Y = ( tf.cast(np.load(yfname),dtype=tf.float32) ).numpy()
dev_set = DataLoader(design_matrices=[("x",X),("y",Y)], batch_size=dev_batch_size, disable_shuffle=True)

def eval_model(agent, dataset, verbose=False):
    """
        Evaluates performance of agent on this fixed-point data sample
    """
    Ly = 0.0 # metric/loss over entire data pool
    Acc = 0.0
    N = 0.0 # number samples seen so far
    for batch in dataset:
        x_name, x = batch[0]
        y_name, y = batch[1]
        N += x.shape[0]

        # simulate inference window
        y_hat, y_count = agent.settle(x, calc_update=False)

        # update tracked fixed-point losses
        Ly = tf.reduce_sum( metric.cat_nll(tf.nn.softmax(y_hat), y) ) + Ly

        # track raw accuracy
        y_ind = tf.cast(tf.argmax(y,1),dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(y_count,1),dtype=tf.int32)
        comp = tf.cast(tf.equal(y_pred,y_ind),dtype=tf.float32)
        Acc += tf.reduce_sum( comp )

        agent.clear()
        if verbose == True:
            print("\r Acc {}  Ly {} over {} samples...".format((Acc/(N * 1.0)), (Ly/(N * 1.0)), N),end="")
    if verbose == True:
        print()
    Ly = Ly / N
    Acc = Acc / N
    return Ly, Acc

################################################################################
# Start simulation
################################################################################
with tf.device(gpu_tag):

    # load in learning curves
    train_curve = np.load("snn/Ly0.npy")
    dev_curve = np.load("snn/vLy0.npy")
    print(" > Generating learning curves...")
    viz.plot_learning_curves(train_curve, dev_curve,
                             plot_fname="snn/mnist_learning_curves.png",
                             y_lab="$-\log p(x)$", x_lab="Epoch")

    agent = io_tools.deserialize(model_fname)
    # re-compile to new batch size
    agent.ngc_model.compile(batch_size=dev_batch_size, use_graph_optim=True)
    print(" > Loading model: ",model_fname)

    ############################################################################
    vLy, vAcc = eval_model(agent, dev_set, verbose=True)
    print(" Ly = {}  Acc = {}".format(vLy, vAcc))

    results_fname = "{}generalization_acc.results".format(out_dir)
    log_t = open(results_fname,"a")
    log_t.write("Generalization on {}:\n".format(xfname))
    log_t.write("  Acc = {} \n  Ly = {} \n".format(vAcc, vLy))
    log_t.close()
