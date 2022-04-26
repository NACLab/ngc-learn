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
#sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# import general simulation utilities
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
import ngclearn.utils.stat_utils as stat
from ngclearn.generator.static.mog import MoG

# import building blocks to construct an NGC system
from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.nodes.enode import ENode
from ngclearn.engine.ngc_graph import NGCGraph
from ngclearn.engine.nodes.fnode import FNode
from ngclearn.engine.proj_graph import ProjectionGraph

seed = 69
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Demo/Tutorial #2 File:
Fits an NGC model to a synthetic dataset of sensory patterns sampled
from a stochastic data generating process based on a mixture of Gaussians (MoG).
With respect to ToD and MSE, we track the prequential estimate (with a fading
factor of 0.99) since this NGC will learn from a streaming data process.

Usage:
$ python sim_train.py --gpu_id=0

@author Alexander Ororbia
################################################################################
"""

def draw_ellipse(position, covariance, ax=None, color=None, **kwargs):
    """Draw an ellipse with a given position/mean and covariance """
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        if covariance.shape == (1, 1):
            width = 2 * np.sqrt(covariance)
            height = width
        else:
            width, height = 2 * np.sqrt(covariance)
    # Draw the Ellipse
    alpha = 0.2
    for nsig in range(1, 4):
        if color is not None:
            ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, alpha=alpha, color=color))
        else:
            ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, alpha=alpha)) #, **kwargs))

def generate_plot(X, mu_list, covar_list, fname, color=None):
    """
    Creates this demo's visualization/plot and saves it to disk.
    """
    Xnpy = X.numpy()
    cmap = matplotlib.cm.get_cmap('plasma')
    if color is None:
        plt.scatter(Xnpy[:, 0], Xnpy[:, 1], cmap='viridis')
    else:
        plt.scatter(Xnpy[:, 0], Xnpy[:, 1], c=color)
    color_selector = 0.0
    color_incr = 1.0/(len(mu_list) * 1.0) #0.3
    for i in range(len(mu_list)):
        mu_i = mu_list[i] #tf.expand_dims(process.mu[i,:],axis=0)
        sigma_i = covar_list[i]
        draw_ellipse(tf.squeeze(mu_i).numpy(), sigma_i.numpy(), color=cmap(color_selector))
        color_selector += color_incr
        #draw_ellipse(tf.squeeze(tf.transpose(gmm.mu[i])).numpy(), tf.transpose(gmm.sigma[i]).numpy()) #, alpha=w * w_factor)
    plt.grid()
    #plt.show()
    plt.savefig(fname) #.format(sample_dir, ptr))
    plt.clf()

################################################################################
# Set up simulated data generating process
################################################################################

mu1 = np.array([[-1.0,1.2]])
cov1 = np.array([[0.1,0.0],
                 [0.0,0.2]])
mu1 = tf.cast(mu1,dtype=tf.float32)
cov1 = tf.cast(cov1,dtype=tf.float32)

mu2 = np.array([[0.85,-1.3]])
cov2 = np.array([[0.2,0.0],
                 [0.0,0.12]])
mu2 = tf.cast(mu2,dtype=tf.float32)
cov2 = tf.cast(cov2,dtype=tf.float32)

mu3 = np.array([[0.92, 1.1]])
cov3 = np.array([[0.12,0.0],
                 [0.0,0.18]])
mu3 = tf.cast(mu3,dtype=tf.float32)
cov3 = tf.cast(cov3,dtype=tf.float32)

mu_list = [mu1, mu2, mu3]
sigma_list = [cov1, cov2, cov3]
process = MoG(means=mu_list, covar=sigma_list, seed=69)
x_dim = mu1.shape[1]


# Set training settings
n_iterations = 400
batch_size = 32 #128 #256

################################################################################
# Set up the NGC system and its ancestral projection co-model
################################################################################

# Build an NGC system with 2 latent variable layers
K = 40
leak = 0.0001 #0.0001
beta = 0.1
z2_dim = 2
z1_dim = 64 #32
integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}
prior_cfg = {"prior_type" : "laplace", "lambda" : 0.0001}

# set up system nodes
z2 = SNode(name="z2", dim=z2_dim, beta=beta, leak=leak, act_fx="identity",
           integrate_kernel=integrate_cfg, prior_kernel=None)
mu1 = SNode(name="mu1", dim=z1_dim, act_fx="identity", zeta=0.0)
e1 = ENode(name="e1", dim=z1_dim)
z1 = SNode(name="z1", dim=z1_dim, beta=beta, leak=leak, act_fx="elu",
           integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)#, lateral_kernel=lateral_cfg)
mu0 = SNode(name="mu0", dim=x_dim, act_fx="identity", zeta=0.0)
e0 = ENode(name="e0", dim=x_dim)
z0 = SNode(name="z0", dim=x_dim, beta=beta, integrate_kernel=integrate_cfg, leak=0.0)

# create cable wiring scheme relating nodes to one another
wght_sd = 0.025 #0.025 #0.05
dcable_cfg = {"type": "dense", "has_bias": False,
              "init" : ("gaussian",wght_sd), "seed" : seed}
pos_scable_cfg = {"type": "simple", "coeff": 1.0}
neg_scable_cfg = {"type": "simple", "coeff": -1.0}

z2_mu1 = z2.wire_to(mu1, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu1.wire_to(e1, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z1.wire_to(e1, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e1.wire_to(z2, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(z2_mu1,"symm_tied")) #anti_symm_tied
#e1.use_mod_factor = use_mod_factor
e1.wire_to(z1, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

z1_mu0 = z1.wire_to(mu0, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu0.wire_to(e0, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z0.wire_to(e0, src_var="phi(z)", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e0.wire_to(z1, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(z1_mu0,"symm_tied")) #anti_symm_tied
e0.wire_to(z0, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

# set up update rules and make relevant edges aware of these
z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"))
z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"))

# Set up graph - execution cycle/order
print(" > Constructing NGC graph")
model = NGCGraph(K=K)
model.proj_update_mag = -1.0 #-1.0
model.proj_weight_mag = 1.0
model.set_cycle(nodes=[z2,z1,z0])
model.set_cycle(nodes=[mu1,mu0])
model.set_cycle(nodes=[e1,e0])
model.apply_constraints()

# build this NGC model's sampling graph
z2_dim = model.getNode("z2").dim
z1_dim = model.getNode("z1").dim
z0_dim = model.getNode("z0").dim
# Set up complementary sampling graph to use in conjunction w/ NGC-graph
s2 = FNode(name="s2", dim=z2_dim, act_fx="identity")
s1 = FNode(name="s1", dim=z1_dim, act_fx="elu")
s0 = FNode(name="s0", dim=z0_dim, act_fx="identity")
s2_s1 = s2.wire_to(s1, src_var="phi(z)", dest_var="dz", point_to_path=z2_mu1)
s1_s0 = s1.wire_to(s0, src_var="phi(z)", dest_var="dz", point_to_path=z1_mu0)
sampler = ProjectionGraph()
sampler.set_cycle(nodes=[s2,s1,s0])

eta = 0.002
eta_v  = tf.Variable( eta ) # set up optimization process
opt = tf.keras.optimizers.Adam(eta_v)

################################################################################
# Fit the NGC model to the stochastic data generating process
################################################################################
def calc_ToD(agent):
    """Measures the total discrepancy (ToD) of a given NGC model"""
    ToD = 0.0
    L1 = agent.extract(node_name="e1", node_var_name="L")
    L0 = agent.extract(node_name="e0", node_var_name="L")
    ToD = -(L0 + L1)
    return ToD

ToD_list = []
x_iter = []
ToD = 0.0
Lx = 0.0
Ns = 0.0
alpha = 0.99 # fading factor
for iter in range(n_iterations):

    x, y = process.sample(n_s=batch_size)
    Ns = x.shape[0] + Ns * alpha

    # conduct iterative inference & update NGC system
    readouts = model.settle(
                    clamped_vars=[("z0","z",x)],
                    readout_vars=[("mu0","phi(z)"),("mu1","phi(z)")]
                )
    x_hat = readouts[0][2]

    ToD = calc_ToD(model) + ToD * alpha # calc ToD
    ToD_list.append((ToD/Ns))
    x_iter.append(iter)
    Lx = tf.reduce_sum( metric.mse(x_hat, x) ) + Lx * alpha
    # update synaptic parameters given current model internal state
    delta = model.calc_updates()
    opt.apply_gradients(zip(delta, model.theta))
    model.apply_constraints()
    model.clear()

    print("\r{} | ToD = {}  MSE = {}".format(iter, ToD/Ns, Lx/Ns), end="")
print()

ToD_vals = np.asarray(ToD_list)
x_iter = np.asarray(x_iter)
fontSize = 20
plt.plot(x_iter, ToD_list, '-', color="purple")
plt.xlabel("Iterations", fontsize=fontSize)
plt.ylabel("ToD", fontsize=fontSize)
plt.grid()
plt.tight_layout()
plt.savefig("tod_curve.jpg")
plt.clf()

################################################################################
# Post-process the model and visualize the latent and sample space
################################################################################
def sample_system(Xs, model, sampler, Ns=-1):
    """
    This routine takes the NGC system and its sampler and produces
    samples of both the latent (z2) and model sample space.
    """
    readouts = model.settle(
                    clamped_vars=[("z0","z",tf.cast(Xs,dtype=tf.float32))],
                    readout_vars=[("mu0","phi(z)"),("z2","z")]
                )
    z2 = readouts[1][2]
    z = z2
    model.clear()
    # estimate latent mode mean and covariance
    z_mu = tf.reduce_mean(z2, axis=0, keepdims=True)
    z_cov = stat.calc_covariance(z2, mu_=z_mu, bias=False)
    z_R = tf.linalg.cholesky(z_cov) # decompose covariance via Cholesky
    if Ns > 0:
        eps = tf.random.normal([Ns, z2.shape[1]], mean=0.0, stddev=1.0, seed=69)
    else:
        eps = tf.random.normal(z2.shape, mean=0.0, stddev=1.0, seed=69)
    # use the re-parameterization trick to sample this mode
    Zs = z_mu + tf.matmul(eps,z_R)
    # now conduct ancestral sampling through the directed generative model
    readouts = sampler.project(
                    clamped_vars=[("s2","z", Zs)],
                    readout_vars=[("s0","phi(z)")]
                )
    X_hat = readouts[0][2]
    sampler.clear()
    # estimate the mean and covariance of the sensory sample space of this mode
    mu_hat = tf.reduce_mean(X_hat, axis=0, keepdims=True)
    sigma_hat = stat.calc_covariance(X_hat, mu_=mu_hat, bias=False)
    return (X_hat, mu_hat, sigma_hat), (z, z_mu, z_cov)

X1,Y1 = process.sample(n_s=64, mode_idx=0)
X2,Y2 = process.sample(n_s=64, mode_idx=1)
X3,Y3 = process.sample(n_s=64, mode_idx=2)
X = tf.concat([X1,X2,X3],axis=0)
Y = tf.concat([Y1,Y2,Y3],axis=0)
generate_plot(X, process.mu, process.sigma, "dataset.jpg", color="black")

# to improve the accuracy of our estimated input space modes, we increase
# the number of samples we draw from our generative model by a factor of 4
samps, lats = sample_system(X1, model, sampler, Ns=256)
X1_hat, x_mu1, x_sig1 = samps
X1_z, X1_z_mu, X1_z_sig = lats

samps, lats = sample_system(X2, model, sampler, Ns=256)
X2_hat, x_mu2, x_sig2 = samps
X2_z, X2_z_mu, X2_z_sig = lats

samps, lats = sample_system(X3, model, sampler, Ns=256)
X3_hat, x_mu3, x_sig3 = samps
X3_z, X3_z_mu, X3_z_sig = lats

# bundle up samples and their statistics and plot
x_mu = [x_mu1, x_mu2, x_mu3]
x_sigma = [x_sig1, x_sig2, x_sig3]
Xs = tf.concat([X1_hat, X2_hat, X3_hat],axis=0)
Xs = Xs.numpy()

# plot the resulting samples
plt.scatter(Xs[:, 0], Xs[:, 1], cmap='viridis')
plt.grid()
plt.savefig("model_samples.jpg")
plt.clf()

generate_plot(tf.cast(X,dtype=tf.float32), x_mu, x_sigma, "model_fit.jpg", color="black")

# bundle up latents and their statistics and plot
z_mu = [X1_z_mu, X2_z_mu, X3_z_mu]
z_sigma = [X1_z_sig, X3_z_sig, X3_z_sig]
Zs = tf.concat([X1_z, X2_z, X3_z],axis=0)
Zs = Zs.numpy()
generate_plot(tf.cast(Zs,dtype=tf.float32), z_mu, z_sigma, "model_latents.jpg", color="red")
