"""
Test the basic/key dynamics of an NGC system constructed with Node(s) and Cable(s)
with synthetic data as a test case.
-------------------------------------------------------------------------------
This code runs the "identity test" for NGC projection and dynamics:
+ create a 3-layer model, w/ tied forward/backward weights and forward weights
  initialized to a diagonal, that performs x |-> y w/ relu activation and
  identity output
+ set x = 1 and clamp it to input of projection graph, should return 1
+ clamp x to ngc graph to input and output/targets states, when run for a
  number of settling steps, should return 1 and remain in equilibrium where
  all internal error nodes are exactly 0.
Repeat the above test for a 3-layer model initialized exactly the same except
with untied forward/backward weights (backward weights initialized to diagonal)

For both model types above, check (after simulating the settling process) that
the synaptic updates for every single weight and bias are exactly matrices of zeros.
-------------------------------------------------------------------------------

The following models are tested:
a 3-layer x->y NGC with tied forward/error weights
a 3-layer x->y NGC with untied forward/error weights

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
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader

from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.nodes.enode import ENode
from ngclearn.engine.ngc_graph import NGCGraph

from ngclearn.engine.nodes.fnode import FNode
from ngclearn.engine.proj_graph import ProjectionGraph

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

# set up parameters of tests

x = tf.ones([1,10])
x_dim = x.shape[1]
z3_dim = x_dim
z2_dim = z3_dim
z1_dim = z2_dim
z0_dim = z1_dim

seed = 69
beta = 1 # must fix step to 1.0 for this test
integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}

print("#######################################################################")
print(" > Testing a proxy NGC graph w/ tied weights")
# set up system nodes
z3 = SNode(name="z3", dim=z3_dim, beta=beta, leak=0, act_fx="identity",
           integrate_kernel=integrate_cfg)
mu2 = SNode(name="mu2", dim=z2_dim, act_fx="identity", zeta=0.0)
e2 = ENode(name="e2", dim=z2_dim)
z2 = SNode(name="z2", dim=z2_dim, beta=beta, leak=0, act_fx="relu",
           integrate_kernel=integrate_cfg)
mu1 = SNode(name="mu1", dim=z1_dim, act_fx="identity", zeta=0.0)
e1 = ENode(name="e1", dim=z1_dim)
z1 = SNode(name="z1", dim=z1_dim, beta=beta, leak=0, act_fx="relu",
           integrate_kernel=integrate_cfg)
mu0 = SNode(name="mu0", dim=z0_dim, act_fx="identity", zeta=0.0)
e0 = ENode(name="e0", dim=z0_dim)
z0 = SNode(name="z0", dim=z0_dim, beta=beta, leak=0.0,
           integrate_kernel=integrate_cfg)

# create cable wiring scheme relating nodes to one another
dcable_cfg = {"type": "dense", "has_bias": True,
              "init" : ("diagonal",1), "seed" : seed} #classic_glorot
pos_scable_cfg = {"type": "simple", "coeff": 1.0}
neg_scable_cfg = {"type": "simple", "coeff": -1.0}

z3_mu2 = z3.wire_to(mu2, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu2.wire_to(e2, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z2.wire_to(e2, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e2.wire_to(z3, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(z3_mu2,"symm_tied"))
e2.wire_to(z2, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

z2_mu1 = z2.wire_to(mu1, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu1.wire_to(e1, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z1.wire_to(e1, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e1.wire_to(z2, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(z2_mu1,"symm_tied"))
e1.wire_to(z1, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

z1_mu0 = z1.wire_to(mu0, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu0.wire_to(e0, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z0.wire_to(e0, src_var="phi(z)", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e0.wire_to(z1, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(z1_mu0,"symm_tied"))
e0.wire_to(z0, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

# set up update rules and make relevant edges aware of these
z3_mu2.set_update_rule(preact=(z3,"phi(z)"), postact=(e2,"phi(z)"))
z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"))
z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"))

# Set up graph - execution cycle/order
ngc_model = NGCGraph(K=10, name="gncn_t1_ffm")
ngc_model.proj_update_mag = -1.0
ngc_model.proj_weight_mag = -1.0
ngc_model.set_cycle(nodes=[z3,z2,z1,z0])
ngc_model.set_cycle(nodes=[mu2,mu1,mu0])
ngc_model.set_cycle(nodes=[e2,e1,e0])
ngc_model.apply_constraints()

# build this NGC model's sampling graph
z3_dim = ngc_model.getNode("z3").dim
z2_dim = ngc_model.getNode("z2").dim
z1_dim = ngc_model.getNode("z1").dim
z0_dim = ngc_model.getNode("z0").dim
# Set up complementary sampling graph to use in conjunction w/ NGC-graph
s3 = FNode(name="s3", dim=z3_dim, act_fx="identity")
s2 = FNode(name="s2", dim=z2_dim, act_fx="relu")
s1 = FNode(name="s1", dim=z1_dim, act_fx="relu")
s0 = FNode(name="s0", dim=z0_dim, act_fx="identity")
s3_s2 = s3.wire_to(s2, src_var="phi(z)", dest_var="dz", point_to_path=z3_mu2)
s2_s1 = s2.wire_to(s1, src_var="phi(z)", dest_var="dz", point_to_path=z2_mu1)
s1_s0 = s1.wire_to(s0, src_var="phi(z)", dest_var="dz", point_to_path=z1_mu0)
sampler = ProjectionGraph()
sampler.set_cycle(nodes=[s3,s2,s1,s0])

# test projection graph
print("----------------")
print("  > Checking ancestral projection graph:")
readouts = sampler.project(
                clamped_vars=[("s3","z",x)],
                readout_vars=[("s0","phi(z)")]
            )
x_sample = readouts[0][2]

print(" => Test for:  x = 1 = s2.z = s2.phi(z)")
s3_z = sampler.extract("s3","z")
s3_phi = sampler.extract("s3","phi(z)")
np.testing.assert_array_equal(x.numpy(), s3_z.numpy())
np.testing.assert_array_equal(x.numpy(), s3_phi.numpy())
print("  PASS!")

print(" => Test for:  x = 1 = s1.z = s1.phi(z)")
s1_z = sampler.extract("s1","z")
s1_phi = sampler.extract("s1","phi(z)")
np.testing.assert_array_equal(x.numpy(), s1_z.numpy())
np.testing.assert_array_equal(x.numpy(), s1_phi.numpy())
print("  PASS!")

print(" => Test for:  x = 1 = s0.z = s0.phi(z)")
s0_z = sampler.extract("s0","z")
s0_phi = sampler.extract("s0","phi(z)")
np.testing.assert_array_equal(x.numpy(), s0_z.numpy())
np.testing.assert_array_equal(x.numpy(), s0_phi.numpy())
print("  PASS!")

print(" => Test for:  x = x_sample")
print("Expected: ",x.numpy())
print("  Output: ",x_sample.numpy())
np.testing.assert_array_equal(x.numpy(), x_sample.numpy())
print("  PASS!")


# test NGC simulation object
print("----------------")
print("  > Checking NGC simulation object:")
readouts = ngc_model.settle(
                clamped_vars=[("z3","z",x),("z0","z",x)],
                readout_vars=[("mu0","phi(z)"),("mu1","phi(z)"),("mu2","phi(z)")]
            )
x_hat = readouts[0][2]

print(" => Test for:  0 = e2.z = e2.phi(z)")
target_value = np.zeros([1,z2_dim])
e2_z = ngc_model.extract("e2","z")
e2_phi = ngc_model.extract("e2","phi(z)")
np.testing.assert_array_equal(target_value, e2_z.numpy())
np.testing.assert_array_equal(target_value, e2_phi.numpy())
print("  PASS!")

print(" => Test for:  0 = e1.z = e1.phi(z)")
target_value = np.zeros([1,z1_dim])
e1_z = ngc_model.extract("e1","z")
e1_phi = ngc_model.extract("e1","phi(z)")
np.testing.assert_array_equal(target_value, e1_z.numpy())
np.testing.assert_array_equal(target_value, e1_phi.numpy())
print("  PASS!")

print(" => Test for:  0 = e0.z = e0.phi(z)")
target_value = np.zeros([1,z0_dim])
e0_z = ngc_model.extract("e0","z")
e0_phi = ngc_model.extract("e0","phi(z)")
np.testing.assert_array_equal(target_value, e0_z.numpy())
np.testing.assert_array_equal(target_value, e0_phi.numpy())
print("  PASS!")

print(" => Test for:  x = x_hat")
print("Expected: ",x.numpy())
print("  Output: ",x_hat.numpy())
np.testing.assert_array_equal(x.numpy(), x_hat.numpy())
print("  PASS!")

print(" => Test for update calculation: all dx should be = 0")
delta = ngc_model.calc_updates()
for i in range(len(delta)):
    target_dx = ngc_model.theta[i] * 0
    dx = delta[i]
    np.testing.assert_array_equal(target_dx.numpy(), dx.numpy())
print("  PASS! (for all {} dx calculations)".format(len(delta)))

print("#######################################################################")

print("#######################################################################")
print(" > Testing a proxy NGC graph w/ untied weights")
# set up system nodes
z3 = SNode(name="z3", dim=z3_dim, beta=beta, leak=0, act_fx="identity",
           integrate_kernel=integrate_cfg)
mu2 = SNode(name="mu2", dim=z2_dim, act_fx="identity", zeta=0.0)
e2 = ENode(name="e2", dim=z2_dim)
z2 = SNode(name="z2", dim=z2_dim, beta=beta, leak=0, act_fx="relu",
           integrate_kernel=integrate_cfg)
mu1 = SNode(name="mu1", dim=z1_dim, act_fx="identity", zeta=0.0)
e1 = ENode(name="e1", dim=z1_dim)
z1 = SNode(name="z1", dim=z1_dim, beta=beta, leak=0, act_fx="relu",
           integrate_kernel=integrate_cfg)
mu0 = SNode(name="mu0", dim=z0_dim, act_fx="identity", zeta=0.0)
e0 = ENode(name="e0", dim=z0_dim)
z0 = SNode(name="z0", dim=z0_dim, beta=beta, leak=0.0,
           integrate_kernel=integrate_cfg)

# create cable wiring scheme relating nodes to one another
z3_mu2 = z3.wire_to(mu2, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu2.wire_to(e2, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z2.wire_to(e2, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e2.wire_to(z3, src_var="phi(z)", dest_var="dz_bu", cable_kernel=dcable_cfg)
e2.wire_to(z2, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

z2_mu1 = z2.wire_to(mu1, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu1.wire_to(e1, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z1.wire_to(e1, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e1.wire_to(z2, src_var="phi(z)", dest_var="dz_bu", cable_kernel=dcable_cfg)
e1.wire_to(z1, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

z1_mu0 = z1.wire_to(mu0, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
mu0.wire_to(e0, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
z0.wire_to(e0, src_var="phi(z)", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
e0.wire_to(z1, src_var="phi(z)", dest_var="dz_bu", cable_kernel=dcable_cfg)
e0.wire_to(z0, src_var="phi(z)", dest_var="dz_td", cable_kernel=neg_scable_cfg)

# set up update rules and make relevant edges aware of these
z3_mu2.set_update_rule(preact=(z3,"phi(z)"), postact=(e2,"phi(z)"))
z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"))
z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"))

# Set up graph - execution cycle/order
ngc_model = NGCGraph(K=10, name="gncn_t1_ffm")
ngc_model.proj_update_mag = -1.0
ngc_model.proj_weight_mag = -1.0
ngc_model.set_cycle(nodes=[z3,z2,z1,z0])
ngc_model.set_cycle(nodes=[mu2,mu1,mu0])
ngc_model.set_cycle(nodes=[e2,e1,e0])
ngc_model.apply_constraints()

print("----------------")
print("  > Checking NGC simulation object:")
readouts = ngc_model.settle(
                clamped_vars=[("z3","z",x),("z0","z",x)],
                readout_vars=[("mu0","phi(z)"),("mu1","phi(z)"),("mu2","phi(z)")]
            )
x_hat = readouts[0][2]

print(" => Test for:  0 = e2.z = e2.phi(z)")
target_value = np.zeros([1,z2_dim])
e2_z = ngc_model.extract("e2","z")
e2_phi = ngc_model.extract("e2","phi(z)")
np.testing.assert_array_equal(target_value, e2_z.numpy())
np.testing.assert_array_equal(target_value, e2_phi.numpy())
print("  PASS!")

print(" => Test for:  0 = e1.z = e1.phi(z)")
target_value = np.zeros([1,z1_dim])
e1_z = ngc_model.extract("e1","z")
e1_phi = ngc_model.extract("e1","phi(z)")
np.testing.assert_array_equal(target_value, e1_z.numpy())
np.testing.assert_array_equal(target_value, e1_phi.numpy())
print("  PASS!")

print(" => Test for:  0 = e0.z = e0.phi(z)")
target_value = np.zeros([1,z0_dim])
e0_z = ngc_model.extract("e0","z")
e0_phi = ngc_model.extract("e0","phi(z)")
np.testing.assert_array_equal(target_value, e0_z.numpy())
np.testing.assert_array_equal(target_value, e0_phi.numpy())
print("  PASS!")

print(" => Test for:  x = x_hat")
print("Expected: ",x.numpy())
print("  Output: ",x_hat.numpy())
np.testing.assert_array_equal(x.numpy(), x_hat.numpy())
print("  PASS!")

print(" => Test for update calculation: all dx should be = 0")
delta = ngc_model.calc_updates()
for i in range(len(delta)):
    target_dx = ngc_model.theta[i] * 0
    dx = delta[i]
    np.testing.assert_array_equal(target_dx.numpy(), dx.numpy())
print("  PASS! (for all {} dx calculations)".format(len(delta)))

print("#######################################################################")
