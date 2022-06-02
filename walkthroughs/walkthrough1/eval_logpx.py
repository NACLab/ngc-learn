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

import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

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
Evaluates/estimates the marginal log likelihood -- log p(x) -- of a
(pre-)trained NGC model given a data sample/pool, i.e., the MNIST test set.

Usage:
$ python eval_logpx.py --config=/path/to/analyze.cfg --gpu_id=0

@author Alexander Ororbia
################################################################################
"""

# GPU arguments
# read in configuration file and extract necessary variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["config=","gpu_id="])
# Collect arguments from argv
n_trials = 1
use_gpu = False
gpu_id = -1
xfname = None
out_dir = None
rotNeg90 = False
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


args = Config(cfg_fname)
xfname = args.getArg("dev_xfname")
out_dir = args.getArg("out_dir")
#n_trials = int(args.getArg("n_trials"))
rotNeg90 = (args.getArg("rotNeg90").lower() == 'true')
results_fname = args.getArg("results_fname")

model_fname = args.getArg("model_fname")
latents_fname = args.getArg("latents_fname")
gmm_fname = args.getArg("gmm_fname")

do_gmm_plot = True

delimiter = "\t"
X = transform.binarize( tf.cast(np.load(xfname),dtype=tf.float32) )#.numpy()
print(" X.shape = ",X.shape)

# data meta-parameters --> 2000 x 5 yields test set of 10,000
n_mc_batches = 2000 #200 # number of Monte Carlo sampled batches
mc_batch_size = 5 #10 #25 #10 #50 #20
n_mc = 5000 #5000 #6000 # number of Monte Carlo samples




def get_n_comp(gmm, use_sklearn=True):
    if use_sklearn is True:
        return gmm.n_components
    else:
        return gmm.k

def sample_gmm(gmm, n_samps, use_sklearn=True):
    if use_sklearn is True:
        np_samps, np_labs = gmm.sample(n_samps)
        z_samp = tf.cast(np_samps,dtype=tf.float32)
    else:
        z_samp, z_labs = gmm.sample(n_samps)
        np_labs = tf.squeeze(z_labs).numpy()
    y_s = tf.one_hot(np_labs, get_n_comp(gmm, use_sklearn=use_sklearn))
    return z_samp, y_s

def estimate_logpx_batch(data, num_samples, model, sampler, use_sklearn=False, binarize_x=True):
    """
        Calculate Monte Carlo sample
        \log p(x) = E_p[p(x|z)] = \log(\int p(x|z) p(z) dz)
                 ~= \log(1/n * \sum_i p(x|z_i) )
                  = \log p(x) = \log(1/n * \sum_i e^{\log p(x|z_i)})
                  = \log p(x) = -\logn + \logsumexp_i(\log p(x|z_i)) // allows us to use logsumexp trick
    """
    top_dim = model.ngc_sampler.getNode("s3").dim # model.z_dims[len(model.z_dims)-1]
    result = []
    for i in range(len(data)):
        # if sampler is None:
        #     #if model.name == "gan":
        #     #    z_samples = sample_uniform(num_samples, top_dim)
        #     #else:
        #     z_samples = stat.sample_gaussian(num_samples, tf.zeros([1,top_dim]), 1.0)
        #     y_s = None # ?????????? tf.random.categorical(pi, num_samples=1)
        # else:
        #z_samples = sample_gmm(sampler, num_samples, use_sklearn=use_sklearn)
        logpz = None
        z_samples, y_s = sample_gmm(sampler, num_samples, use_sklearn=use_sklearn)

        zs_dim = z_samples.shape[1]
        delta = zs_dim - top_dim
        if delta > 0:
            zs = z_samples[:,0:top_dim]
            ys = z_samples[:,top_dim:zs_dim]
            y_s = ys
            z_samples = zs

        datum = tf.expand_dims(tf.cast(data[i],dtype=tf.float32),axis=0)
        x_predict = agent.project(z_samples)

        clip_offset = 1e-6 #0.0001
        x_predict = tf.clip_by_value(x_predict, clip_offset, 1. - clip_offset)

        # \log p(x|z) = Binary cross entropy
        #logp_xz = np.sum(datum * np.log(x_predict) + (1. - datum) * np.log(1.0 - x_predict), axis=1)
        if binarize_x is True or use_bern_cross_entropy is True:
            # binary data - calc Bernoulli log likelihood
            logp_xz = tf.reduce_sum(datum * tf.math.log(x_predict) + (1.0 - datum) * tf.math.log(1.0 - x_predict), axis=1, keepdims=True)
        else:
            # continuous data - calc Gaussian log likelihood with mu and fixed unit variance
            sigma = 1.0 # std dev is assumed to be fixed at 1.0
            mu = x_predict
            diff = (datum - mu)
            # n is assumed = 1.0
            logp_xz = tf.reduce_sum(-(diff * diff) * (1.0/(2.0 * sigma)) - tf.math.log(2.0 * np.pi) * 0.5, axis=1, keepdims=True)

        argsum = logp_xz #+ logpz
        logpx = tf.math.log(num_samples * 1.0) + tf.reduce_logsumexp(argsum)
        #logpx = -np.log(num_samples) + logsumexp(argsum)
        result.append(logpx)

    return np.array(result)

def estimate_logpx(data_, num_samples, model, sampler=None, n_mc_batches=5, verbosity=0, batch_size=20):
    data = data_.numpy()
    batches = []
    iterations = int(np.ceil(1. * len(data) / batch_size))
    for b in range(iterations):
        batch_data = data[b * batch_size:(b+1) * batch_size]
        batches.append(estimate_logpx_batch(batch_data, num_samples, model, sampler))
        #if verbosity and b % max(11 - verbosity, 1) == 0:
        str = "Batch %d [%d, %d): %.2f" % (b, b*batch_size, (b+1) * batch_size,
                                           np.mean(np.concatenate(batches)))
        print("\r{0}".format(str),end="")
        #print("Batch %d [%d, %d): %.2f" % (b, b*batch_size, (b+1) * batch_size,
        #                                   np.mean(np.concatenate(batches))))
        if b >= n_mc_batches:
            break
        #np.mean(np.concatenate(batches))
    print()
    return np.mean(np.concatenate(batches))

with tf.device(gpu_tag):
    print(" >> Loading pre-trained model: {0}".format(model_fname))
    agent = io_utils.deserialize(model_fname)
    sampler = io_utils.deserialize(gmm_fname)

    if do_gmm_plot is True:  # sample from density estimator
        sample_plot_fname = "{}samples.png".format(out_dir)
        n_s = 1 #3 # number of rows
        n_a = 10 #5 # number of columns

        px = py = int(np.sqrt(X.shape[1])) #28
        # sample using GMM learned density
        n_samp_steps = 1
        z_in = None
        for nn in range(n_samp_steps):
            #z_samples, y_s = sample_gmm(sampler, num_samples, use_sklearn=False)
            z_in, y_in = sample_gmm(sampler, (n_s * n_a), use_sklearn=False)
        #_, samples = sampler.decode(z_in,y=y_in,apply_post=True)
        print(z_in.shape)
        samples = agent.project(z_in)
        # clip_offset = 1e-6
        # samples = tf.clip_by_value(samples, clip_offset, 1 - clip_offset)

        #samples = tf.cast(tf.greater(samples, 0.5), dtype=tf.float32)
        #model.clear_var_history()
        # smask = tf.cast(tf.greater(samples, 0.025),dtype=tf.float32)
        # samples = samples * smask

        # shuffle order of rows in sample matrix for visual variety
        n_samples = samples.numpy()
        np.random.shuffle(n_samples)
        samples = tf.cast(n_samples, dtype=tf.float32)

        #nx = ny = n_a
        nx = n_s
        ny = n_a
        px_dim = px
        py_dim = py
        canvas = np.empty((px_dim*nx, py_dim*ny))
        ptr = 0
        for i in range(0,nx,1):
            for j in range(0,ny,1):
                xs = tf.expand_dims(tf.cast(samples[ptr,:],dtype=tf.float32),axis=0)
                xs = xs.numpy() #tf.make_ndarray(x_mean)
                xs = xs[0].reshape(px_dim, py_dim)
                if rotNeg90 is True:
                    xs = np.rot90(xs, -1)
                canvas[(nx-i-1)*px_dim:(nx-i)*px_dim, j*py_dim:(j+1)*py_dim] = xs
                ptr += 1
        plt.figure(figsize=(8, 10))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.tight_layout()
        plt.axis('off')
        #print(" SAVE: {0}{1}".format(out_dir,"gmm_decoded_samples.jpg"))
        plt.savefig("{0}".format(sample_plot_fname), bbox_inches='tight', pad_inches=0)
        plt.clf()

    batch_size = mc_batch_size #20 # 50
    logpx = estimate_logpx(X, num_samples=n_mc, model=agent, sampler=sampler,
                           n_mc_batches=n_mc_batches, verbosity=1, batch_size=batch_size)
    print(" -> Final Measurement/Stat:")
    print("      log p(x) = %.2f" % logpx)
    #log_px_list.append(logpx)

    results_fname = "{}logpx.results".format(out_dir)
    log_t = open(results_fname,"a")
    log_t.write("Likelihood Test:\n")
    log_t.write("  log[p(x)] = {0} \n".format(logpx))
    log_t.close()
