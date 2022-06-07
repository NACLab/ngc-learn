import sys, getopt, optparse
import pickle
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
from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.nodes.spiking.spnode_lif import SpNode_LIF
from ngclearn.engine.nodes.spiking.spnode_enc import SpNode_Enc
from ngclearn.engine.nodes.enode import ENode
from ngclearn.engine.nodes.fnode_ba import FNode_BA
from ngclearn.engine.ngc_graph import NGCGraph

class SNN_BA:
    """
    A spiking neural network (SNN) classifier that adapts its synaptic cables
    via broadcast alignment. Specifically, this model is a generalization of
    the one proposed in:

    Samadi, Arash, Timothy P. Lillicrap, and Douglas B. Tweed. "Deep learning with
    dynamic spiking neurons and fixed feedback weights." Neural computation 29.3
    (2017): 578-602.

    This model encodes its real-valued inputs as Poisson spike trains with
    spikes emitted at a rate of approximately 63.75 Hz. The internal nodes
    and output nodes operate under the leaky integrate-and-fire spike response
    model and operate with a relative refractory rate of 1.0 ms. The integration
    time constant for this model has been set to 0.25 ms.

    | Node Name Structure:
    | z2 -(z2-mu1)-> mu1 ; z1 -(z1-mu0-)-> mu0 ;e0; z0
    | e0 -> d1 and z1 -> d1, where d1 is a teaching signal for z1
    |  Note that z2 = x and z0 = y, yielding a classifier

    Args:
        args: a Config dictionary containing necessary meta-parameters for the SNN-BA

    | DEFINITION NOTE:
    | args should contain values for the following:
    | * batch_size - the fixed batch-size to be fed into this model
    | * z_dim - # of latent variables in layers z1
    | * x_dim - # of latent variables in layer z2 or sensory x
    | * y_dim - # of variables in layer z0 or target y
    | * seed - number to control determinism of weight initialization
    | * wght_sd - standard deviation of Gaussian initialization of weights (optional)
    | * T - # of time steps to take when conducting iterative settling (if not online)
    """
    def __init__(self, args):
        self.args = args

        self.T = 100 # 50
        self.T_prime = 20 # "burn-in" period before learning is conducted
        if self.args.hasArg("T") == True:
            self.T = int(self.args.getArg("T"))
        batch_size = int(self.args.getArg("batch_size"))
        z_dim = int(self.args.getArg("z_dim"))
        x_dim = int(self.args.getArg("x_dim"))
        y_dim = int(self.args.getArg("y_dim"))
        seed = int(self.args.getArg("seed"))
        wght_sd = 0.055
        if self.args.hasArg("wght_sd") == True:
            wght_sd = float(self.args.getArg("wght_sd"))

        dt = 0.25 # this is used #1e-3
        tau_mem = 20 # this is used
        leak = 0 # unused/not relevant
        beta = 0.1 # unused/not relevant
        V_thr = 0.4 # this is used #1
        R = 5.1 # unused # 1
        integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True, "dt" : dt}
        #spike_kernel = {"V_thr" : V_thr, "R" : R, "C" : 5e-3, "tau_curr" : 2e-3}
        spike_kernel = {"V_thr" : V_thr, "tau_m" : tau_mem, "ref_T" : 1.0}#, "A_theta" : 0.25, "tau_A" : 20.0}
        trace_kernel = {"dt" : dt, "tau_trace" : 5.0}

        # set up system nodes
        z2 = SpNode_Enc(name="z2", dim=x_dim, gain=0.25, trace_kernel=trace_kernel)
        #z2 = SNode(name="z2", dim=x_dim, beta=beta, leak=0.0, zeta=0.0)
        mu1 = SNode(name="mu1", dim=z_dim, act_fx="identity", zeta=0.0)
        z1 = SpNode_LIF(name="z1", dim=z_dim, integrate_kernel=integrate_cfg,
                        spike_kernel=spike_kernel, trace_kernel=trace_kernel)
        mu0 = SNode(name="mu0", dim=y_dim, act_fx="identity", zeta=0.0)
        z0 = SpNode_LIF(name="z0", dim=y_dim, integrate_kernel=integrate_cfg,
                        spike_kernel=spike_kernel, trace_kernel=trace_kernel)
        e0 = ENode(name="e0", dim=y_dim)
        t0 = SNode(name="t0", dim=y_dim, beta=beta, integrate_kernel=integrate_cfg, leak=0.0)
        d1 = FNode_BA(name="d1", dim=z_dim, act_fx="identity")

        # create cable wiring scheme relating nodes to one another
        init_kernels = {"A_init" : ("gaussian", wght_sd), "b_init" : ("zeros",)}
        dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : seed}
        pos_scable_cfg = {"type": "simple", "coeff": 1.0}
        neg_scable_cfg = {"type": "simple", "coeff": -1.0}

        z2_mu1 = z2.wire_to(mu1, src_comp="Sz", dest_comp="dz_td", cable_kernel=dcable_cfg,
                            short_name="W2")
        mu1.wire_to(z1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=pos_scable_cfg,
                    short_name="1")
        z1_mu0 = z1.wire_to(mu0, src_comp="Sz", dest_comp="dz_td", cable_kernel=dcable_cfg,
                            short_name="W1")
        mu0.wire_to(z0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=pos_scable_cfg,
                    short_name="1")
        z0.wire_to(e0, src_comp="Sz", dest_comp="pred_mu", cable_kernel=pos_scable_cfg,
                   short_name="1")
        t0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg,
                   short_name="1")
        e0.wire_to(t0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg,
                   short_name="-1")

        z1.wire_to(d1, src_comp="Jz", dest_comp="Jz", cable_kernel=pos_scable_cfg,
                   short_name="1")
        e0_z1 = e0.wire_to(d1, src_comp="phi(z)", dest_comp="dz", cable_kernel=dcable_cfg,
                           short_name="B1")

        # set up update rules and make relevant edges aware of these
        from ngclearn.engine.cables.rules.hebb_rule import HebbRule

        rule1 = HebbRule() # create a local weighted Hebbian rule for internal layer
        rule1.set_terms(terms=[(z2,"Sz"), (d1,"phi(z)")], weights=[1.0, (1.0/(x_dim * 1.0))])
        z2_mu1.set_update_rule(update_rule=rule1, param=["A", "b"])

        rule2 = HebbRule() # create a local weighted Hebbian rule for output layer
        rule2.set_terms(terms=[(z1,"Sz"), (e0,"phi(z)")], weights=[1.0, (1.0/(z_dim * 1.0))])
        z1_mu0.set_update_rule(update_rule=rule2, param=["A", "b"])

        # Set up graph - execution cycle/order
        model = NGCGraph(name="snn_ba")
        model.set_cycle(nodes=[z2, mu1, z1, mu0, z0, t0])
        model.set_cycle(nodes=[e0])
        model.set_cycle(nodes=[d1])
        info = model.compile(batch_size=batch_size)

        self.ngc_model = model
        self.opt = tf.keras.optimizers.SGD(1.0) # this SNN updates each time step w/ SGD and lr = 1

    def predict(self, x):
        """
        Predicts the target for a given *x*. Specifically, this function
        will return spike counts, one per class in *y* -- taking the
        argmax of these counts will yield the model's predicted label.

        Args:
            z_sample: the input sample to project through the NGC graph

        Returns:
            y_sample (spike counts from the underlying predictive model)
        """
        y_hat, y_count = self.settle(x, calc_update=False)
        return y_count

    def settle(self, x, y=None, calc_update=True):
        """
        Run an iterative settling process to find latent states given clamped
        input and output variables, specifically simulating the dynamics of
        the spiking neurons internal to this SNN model.
        Note that this functions returns two outputs -- the first is a count
        matrix (each row is a sample in mini-batch) and each column represents
        the count for one class in y, and the second is an approximate
        probability distribution computed as a softmax over an average across
        the electrical currents produced at each step of simulation.

        Args:
            x: sensory input to clamp top-most layer (z2) to

            y: target output activity, i.e., label target

            calc_update: if True, computes synaptic updates @ end of settling
                process (Default = True)

        Returns:
            y_count (spike counts per class in y), y_hat (approximate probability
                distribution for y)
        """
        y_ = y
        if y_ is None: # generate an empty label vector if none provided
            y_ = tf.zeros([x.shape[0],self.ngc_model.getNode("z0").dim])
        y_hat = 0.0
        self.ngc_model.set_to_resting_state()
        y_count = y_ * 0
        learn_synapses = False
        for t in range(self.T): # sum statistics over time
            if t >= self.T_prime:
                if y is not None: # if no label provided, then no learning...
                    learn_synapses = calc_update
            self.ngc_model.clamp([("z2", "z", x), ("t0", "z", y_)])
            delta = self.ngc_model.step(calc_delta=learn_synapses)
            y_hat = self.ngc_model.extract("z0", "Jz") + y_hat
            y_count += self.ngc_model.extract("z0", "Sz")

            if delta is not None:
                self.opt.apply_gradients(zip(delta, self.ngc_model.theta))
        self.delta = delta
        self.ngc_model.clear()
        y_hat = tf.nn.softmax(y_hat/self.T)
        return y_hat, y_count

    def clear(self):
        """Clears the states/values of the stateful nodes in this NGC system"""
        self.ngc_model.clear()
        self.delta = None
