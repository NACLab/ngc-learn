import os
import sys
import copy
#from config import Config
import tensorflow as tf
import numpy as np

from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.nodes.enode import ENode
from ngclearn.engine.ngc_graph import NGCGraph

from ngclearn.engine.nodes.fnode import FNode
from ngclearn.engine.proj_graph import ProjectionGraph

from ngclearn.utils.io_utils import parse_simulation_info

class GNCN_t1_ISTA:
    """
    A demonstration model to illustrate how to build an NGC structure that is
    compliant with the basic functional requirements of the ngc-learn
    Model Museum. (This code is meant to accompany/be used for
    Demonstration #5 in the ngc-learn docs.)

    Note that this model implements a "deep ISTA" model or, rather, an NGC
    system the strictly assumes its latent sparse codes (z1 and z2) must
    be guided to sparse representations using a soft thresholding function.
    This model could be named, under the naming convention of (Ororbia & Kifer 2022),
    as GNCN-t1/ISTA.

    | Node Name Structure:
    | z2 -(z2-mu1)-> mu1 ;e1; z1 -(z1-mu0-)-> mu0 ;e0; z0
    | Soft-thresholding function applied over z2 and z1

    Args:
        args: a Config dictionary containing necessary meta-parameters for the GNCN-t1

    | DEFINITION NOTE:
    | args should contain values for the following:
    | * batch_size - the fixed batch-size to be fed into this model
    | * z_top_dim - # of latent variables in layer z2 (top-most layer)
    | * z_dim - # of latent variables in layers z1
    | * x_dim - # of latent variables in layer z0 or sensory x
    | * seed - number to control determinism of weight initialization
    | * wght_sd - standard deviation of Gaussian initialization of weights
    | * beta - latent state update factor
    | * leak - strength of the leak variable in the latent states
    | * threshold - type of threshold to use (Default = "none")
    | * thr_lmbda - strength of the threshold applied over latent state activities
        (only if threshold != "none")
    | * K - # of steps to take when conducting iterative inference/settling
    | * act_fx - activation function for layers z1 and z2
    | * out_fx - activation function for layer mu0 (prediction of z0) (Default: sigmoid)

    """
    def __init__(self, args):
        self.args = args

        batch_size = int(self.args.getArg("batch_size"))
        z_top_dim = int(self.args.getArg("z_top_dim"))
        z_dim = int(self.args.getArg("z_dim"))
        x_dim = int(self.args.getArg("x_dim"))

        seed = int(self.args.getArg("seed")) #69
        beta = float(self.args.getArg("beta"))
        K = int(self.args.getArg("K"))
        act_fx = self.args.getArg("act_fx") #"tanh"
        out_fx = "identity"
        if self.args.hasArg("out_fx") == True:
            out_fx = self.args.getArg("out_fx")
        leak = float(self.args.getArg("leak")) #0.0

        thr_lbmda = 0.0
        thr_cfg = None
        if self.args.hasArg("threshold") == True:
            threshold = self.args.getArg("threshold")
            thr_lbmda = float(self.args.getArg("thr_lambda"))
            thr_cfg = {"threshold_type" : threshold, "thr_lambda" : thr_lbmda}

        # set up state integration function
        integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}
        constraint_cfg = {"clip_type":"forced_norm_clip","clip_mag":1.0,"clip_axis":1}

        # set up system nodes
        z2 = SNode(name="z2", dim=z_top_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, threshold_kernel=thr_cfg)
        mu1 = SNode(name="mu1", dim=z_dim, act_fx="identity", zeta=0.0)
        e1 = ENode(name="e1", dim=z_dim)
        z1 = SNode(name="z1", dim=z_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, threshold_kernel=thr_cfg)
        mu0 = SNode(name="mu0", dim=x_dim, act_fx=out_fx, zeta=0.0)
        e0 = ENode(name="e0", dim=x_dim)
        z0 = SNode(name="z0", dim=x_dim, beta=beta, integrate_kernel=integrate_cfg, leak=0.0)

        # create cable wiring scheme relating nodes to one another
        dcable_cfg = {"type": "dense", "init" : ("unif_scale",1.0), "seed" : seed}
        pos_scable_cfg = {"type": "simple", "coeff": 1.0}
        neg_scable_cfg = {"type": "simple", "coeff": -1.0}



        z2_mu1 = z2.wire_to(mu1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
        z2_mu1.set_constraint(constraint_cfg)
        mu1.wire_to(e1, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
        z1.wire_to(e1, src_comp="z", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
        e1.wire_to(z2, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z2_mu1,"A^T"))
        e1.wire_to(z1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg)

        z1_mu0 = z1.wire_to(mu0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
        z1_mu0.set_constraint(constraint_cfg)
        mu0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
        z0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
        e0.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z1_mu0,"A^T"))
        e0.wire_to(z0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg)

        # set up lateral recurrent connections
        #z2_to_z2 = z2.wire_to(z2, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z2_mu1,"A*A^T"))
        #z1_to_z1 = z1.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z1_mu0,"A*A^T"))

        # set up update rules and make relevant edges aware of these
        z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"))
        z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"))

        # Set up graph - execution cycle/order
        print(" > Constructing NGC graph")
        ngc_model = NGCGraph(K=K, name="gncn_t1_ista")
        ngc_model.set_cycle(nodes=[z2,z1,z0])
        ngc_model.set_cycle(nodes=[mu1,mu0])
        ngc_model.set_cycle(nodes=[e1,e0])
        ngc_model.apply_constraints()
        info = ngc_model.compile(batch_size=batch_size)
        self.info = parse_simulation_info(info)
        self.ngc_model = ngc_model


        # set up this NGC model's initialization graph
        inf_constraint_cfg = {"clip_type":"norm_clip","clip_mag":1.0,"clip_axis":0}
        z2_dim = ngc_model.getNode("z2").dim
        z1_dim = ngc_model.getNode("z1").dim
        z0_dim = ngc_model.getNode("z0").dim

        s0 = FNode(name="s0", dim=z0_dim, act_fx=out_fx)
        s1 = FNode(name="s1", dim=z1_dim, act_fx=act_fx)
        st1 = FNode(name="st1", dim=z1_dim, act_fx="identity")
        s2 = FNode(name="s2", dim=z2_dim, act_fx=act_fx)
        st2 = FNode(name="st2", dim=z2_dim, act_fx="identity")
        s0_s1 = s0.wire_to(s1, src_comp="phi(z)", dest_comp="dz", cable_kernel=dcable_cfg)
        s0_s1.set_constraint(inf_constraint_cfg)
        s1_s2 = s1.wire_to(s2, src_comp="phi(z)", dest_comp="dz", cable_kernel=dcable_cfg)
        s1_s2.set_constraint(inf_constraint_cfg)

        e1_inf = ENode(name="e1_inf", dim=z_dim)
        s1.wire_to(e1_inf, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
        st1.wire_to(e1_inf, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
        e2_inf = ENode(name="e2_inf", dim=z_dim)
        s2.wire_to(e2_inf, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
        st2.wire_to(e2_inf, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)

        # set up update rules and make relevant edges aware of these
        s0_s1.set_update_rule(preact=(s0,"phi(z)"), postact=(e1_inf,"phi(z)"))
        s1_s2.set_update_rule(preact=(s1,"phi(z)"), postact=(e2_inf,"phi(z)"))

        sampler = ProjectionGraph()
        sampler.set_cycle(nodes=[s0,s1,s2])
        sampler.set_cycle(nodes=[st1,st2])
        sampler.set_cycle(nodes=[e1_inf,e2_inf])
        sampler_info = sampler.compile()
        self.sampler_info = parse_simulation_info(sampler_info)
        self.ngc_sampler = sampler

        # synaptic adjustments for the NGC model
        self.delta = None
        # synaptic adjustments for the inference/initialization model
        self.s_delta = None

    def encode(self, x_sample):
        """
        Run projection scheme to get initial encoding of "x_sample".

        Args:
            x_sample: the input sample to project through the NGC graph

        Returns:
            z_sample (sample(s) of the underlying generative model latent space)
        """
        readouts = self.ngc_sampler.project(
                        clamped_vars=[("s0","z",tf.cast(x_sample,dtype=tf.float32))],
                        readout_vars=[("s2","phi(z)")]
                    )
        z_sample = readouts[0][2]
        return z_sample

    def settle(self, x, calc_update=True):
        """
        Run an iterative settling process to find latent states given clamped
        input and output variables. NOTE that this function employs
        amortized inference -- the NGC system's settling process is initialized
        from the state of the inference projection graph.

        Args:
            x: sensory input to reconstruct/predict

            calc_update: if True, computes synaptic updates @ end of settling
                process for both NGC system and inference co-model (Default = True)

        Returns:
            x_hat (predicted x)
        """
        # run recognition model
        readouts = self.ngc_sampler.project(
                        clamped_vars=[("s0","z",tf.cast(x,dtype=tf.float32))],
                        readout_vars=[("s1","z"),("s2","z")]
                    )
        s1 = readouts[0][2]
        s2 = readouts[1][2]
        # now run the settling process
        readouts, delta = self.ngc_model.settle(
                            clamped_vars=[("z0","z", x)],
                            init_vars=[("z1","z",s1),("z2","z",s2)],
                            readout_vars=[("mu0","phi(z)"),("z1","z"),
                                          ("z2","z")],
                            calc_delta=calc_update
                          )
        self.delta = delta # store delta to constructor for later retrieval

        if calc_update == True:
            # now compute updates to recognition model given current state of system
            z1 = readouts[1][2]
            z2 = readouts[2][2]
            self.ngc_sampler.project(
                clamped_vars=[("s0","z",tf.cast(x,dtype=tf.float32)),
                              ("s1","z",s1),("s2","z",s2),
                              ("st1","z",z1),("st2","z",z2)]
            )
            self.s_delta = self.ngc_sampler.calc_updates()

        x_hat = readouts[0][2]
        return x_hat

    def calc_updates(self, avg_update=True, decay_rate=-1.0): # decay_rate=0.001
        """
        Calculate adjustments to parameters under this given model and its
        current internal state values

        Returns:
            delta, a list of synaptic updates (that follow order of ngc_model.theta) AND
            s_delta,  a list of synaptic updates (that follow order of ngc_sampler.theta)
        """
        Ns = self.ngc_model.extract("z0","phi(z)").shape[0]
        delta = self.delta
        if avg_update is True:
            for p in range(len(delta)):
                delta[p] = delta[p] * (1.0/(Ns * 1.0))
                if decay_rate > 0.0: # weight decay
                    delta[p] = delta[p] - (self.ngc_model.theta[p] * decay_rate)
        s_delta = self.s_delta
        if avg_update is True:
            for p in range(len(delta)):
                s_delta[p] = s_delta[p] * (1.0/(Ns * 1.0))
                if decay_rate > 0.0: # weight decay
                    s_delta[p] = s_delta[p] - (self.ngc_sampler.theta[p] * decay_rate)
        return delta, s_delta

    def clear(self):
        """Clears the states/values of the stateful nodes in this NGC system"""
        self.ngc_model.clear()
        self.ngc_sampler.clear()
        self.delta = None

    def print_norms(self):
        """Prints the Frobenius norms of each parameter of this system"""
        str = ""
        for param in self.ngc_model.theta:
            str = "{} | {} : {}".format(str, param.name, tf.norm(param,ord=2))
        #str = "{}\n".format(str)
        return str

    def set_weights(self, source, tau=0.005): #0.001):
        """
        Deep copies weight variables of another model (of the same exact type)
        into this model's weight variables/parameters.

        Args:
            source: the source model to extract/transfer params from

            tau: if > 0, the Polyak averaging coefficient (-1 sets to hard deep copy/transfer)
        """
        #self.param_var = copy.deepcopy(source.param_var)
        if tau >= 0.0:
            for l in range(0, len(self.ngc_model.theta)):
                self.ngc_model.theta[l].assign( self.ngc_model.theta[l] * (1 - tau) + source.ngc_model.theta[l] * tau )
        else:
            for l in range(0, len(self.ngc_model.theta)):
                self.ngc_model.theta[l].assign( source.ngc_model.theta[l] )
