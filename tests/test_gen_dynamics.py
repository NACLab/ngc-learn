"""
Test the basic/key inference and learning dynamics of an NGC directed generative
system constructed with Node(s) and Cable(s), checked against an
analytical/hand-coded NGC model.
-------------------------------------------------------------------------------
This code runs the "ngc dynamics test" for NGC dynamics given real-valued input:
+ create a directed generative NGC model with 2 latent layers, z1 and z2 (no biases)
+ create a hand-coded NGC system with weights set to be equal to those initialized
  in the ngc-learn model above class
+ running both NGC system and hand-coded NGC, should get a Euclidean norm of exactly
  zero for the difference between the values returned by each for each post-activation
  latent layers z1f & z2f, error neuron layers e0 & e1, expectations mu0 & mu1,
  and calculated weight updates dW1 & dW2
The test above is repeated for five iterations (or training steps). Since both
the NGC generative model and the hand-coded NGC object are updated with SGD, they
should NOT diverge and always yield Euclidean norms exactly equal to zero at
each iteration.
-------------------------------------------------------------------------------

The following models are tested:
a 3-layer NGC directed generative model

This (non-exhaustive) test script checks for qualitative irregularities in each
model's behavior and, indirectly, the functioning of ngc-learn's nodes and cables.
Please read the documentations in /docs/ for an overview and description/use of
Nodes and Cables as well as for practical use-cases through the
demonstrations (under the /examples directory).

As this is strictly a qualitative set of tests, it is up to the developer / user
to examine for specific irregularities.
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
import ngclearn.utils.stat_utils as stat
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools

from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.nodes.enode import ENode
from ngclearn.engine.ngc_graph import NGCGraph


seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)


def calc(x, W1, W2, K, beta): # analytical/hand-coded 2-layer NGC
    z2 = tf.zeros([x.shape[0],W2.shape[0]])
    z1 = tf.zeros([x.shape[0],W1.shape[0]])
    z0 = x
    e1 = tf.zeros([x.shape[0],W1.shape[0]])
    e0 = tf.zeros([x.shape[0],x.shape[1]])
    for k in range(K):
        # correct
        d2 = (tf.matmul(e1,W2,transpose_b=True) * transform.d_tanh(z2))
        z2 = z2 + d2 * beta
        z2_f = transform.tanh(z2)
        d1 = (tf.matmul(e0,W1,transpose_b=True) * transform.d_tanh(z1)) - e1
        z1 = z1 + d1 * beta
        # predict and check
        z1_f = transform.tanh(z1)
        z1_mu = tf.matmul(z2_f, W2)
        e1 = z1 - z1_mu
        z0_mu = tf.matmul(z1_f, W1)
        e0 = z0 - z0_mu
    dW2 = tf.matmul(z2_f, e1, transpose_a=True)
    dW2 = -dW2 # flip direction to descent
    dW1 = tf.matmul(z1_f, e0, transpose_a=True)
    dW1 = -dW1 # flip direction to descent
    return z0_mu, z1_mu, z1_f, z2_f, e0, e1, dW1, dW2

# set up parameters of tests

x = stat.sample_uniform(6,5) #tf.ones([1,10])
x_dim = x.shape[1]
z2_dim = 9
z1_dim = 7
z0_dim = x_dim
K = 10

seed = 69
beta = 0.1
integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}

print("#######################################################################")
print(" > Testing a proxy NGC graph w/ tied weights")
# set up system nodes
z2 = SNode(name="z2", dim=z2_dim, beta=beta, leak=0, act_fx="tanh",
           integrate_kernel=integrate_cfg)
mu1 = SNode(name="mu1", dim=z1_dim, act_fx="identity", zeta=0.0)
e1 = ENode(name="e1", dim=z1_dim)
z1 = SNode(name="z1", dim=z1_dim, beta=beta, leak=0, act_fx="tanh",
           integrate_kernel=integrate_cfg)
mu0 = SNode(name="mu0", dim=z0_dim, act_fx="identity", zeta=0.0)
e0 = ENode(name="e0", dim=z0_dim)
z0 = SNode(name="z0", dim=z0_dim, beta=beta, leak=0.0,
           integrate_kernel=integrate_cfg)

# create cable wiring scheme relating nodes to one another
init_kernels = {"A_init" : ("gaussian",0.025)}
dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : seed}
pos_scable_cfg = {"type": "simple", "coeff": 1.0}
neg_scable_cfg = {"type": "simple", "coeff": -1.0}

z2_mu1 = z2.wire_to(mu1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
mu1.wire_to(e1, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
z1.wire_to(e1, src_comp="z", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
e1.wire_to(z2, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z2_mu1,"A^T"))
e1.wire_to(z1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg)

z1_mu0 = z1.wire_to(mu0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
mu0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
z0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
e0.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z1_mu0,"A^T"))
e0.wire_to(z0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg)

# set up update rules and make relevant edges aware of these
z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"), param=["A"])
z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"), param=["A"])

# Set up graph - execution cycle/order
ngc_model = NGCGraph(K=K, name="gncn_t1")
ngc_model.set_cycle(nodes=[z2,z1,z0])
ngc_model.set_cycle(nodes=[mu1,mu0])
ngc_model.set_cycle(nodes=[e1,e0])
ngc_model.set_learning_order([z2_mu1, z1_mu0]) # order: W2, W1
ngc_model.apply_constraints()
info = ngc_model.compile(batch_size=x.shape[0], use_graph_optim=False)

W2 = ngc_model.theta[0] + 0
W1 = ngc_model.theta[1] + 0

opt = tf.keras.optimizers.SGD(0.1)

goal = np.zeros([1,1])

n_iter = 5 #100
for t in range(n_iter):
    mu0, mu1, z1_f, z2_f, e0, e1, dW1, dW2 = calc(x, W1, W2, K, beta)

    readouts, delta = ngc_model.settle(
                        clamped_vars=[("z0","z",x)],
                        readout_vars=[("mu0","phi(z)"),("mu1","phi(z)")]
                      )
    mu0_ = readouts[0][2]
    mu1_ = readouts[1][2]
    z2_f_ = ngc_model.extract("z2", "phi(z)")
    z1_f_ = ngc_model.extract("z1", "phi(z)")
    e1_ = ngc_model.extract("e1", "phi(z)")
    e0_ = ngc_model.extract("e0", "phi(z)")
    dW2_ = delta[0] + 0
    dW1_ = delta[1] + 0

    print("--------------- CHECK on Iteration {} ---------------".format(t))

    print(" => Test for:  NGC.z2.phi(z) = Analytical.z2.phi(z)")
    check = tf.norm(z2_f_-z2_f)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.z1.phi(z) = Analytical.z1.phi(z)")
    check = tf.norm(z1_f_-z1_f)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.e1.phi(z) = Analytical.e1.phi(z)")
    check = tf.norm(e1_-e1)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.e0.phi(z) = Analytical.e0.phi(z)")
    check = tf.norm(e0_-e0)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.mu1.phi(z) = Analytical.mu1.phi(z)")
    check = tf.norm(mu1_-mu1)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.mu0.phi(z) = Analytical.mu0.phi(z)")
    check = tf.norm(mu0_-mu0)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.dW2 = Analytical.dW2")
    check = tf.norm(dW2_-dW2)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    print(" => Test for:  NGC.dW1 = Analytical.dW1")
    check = tf.norm(dW1_-dW1)
    np.testing.assert_array_equal(goal, check.numpy())
    print("  PASS!")

    opt.apply_gradients(zip(delta, ngc_model.theta))
    ngc_model.clear()

    W2 = ngc_model.theta[0] + 0
    W1 = ngc_model.theta[1] + 0
