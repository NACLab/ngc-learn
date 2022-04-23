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

class GNCN_PDH:
    """
    Structure for constructing the model proposed in:

    Ororbia, Alexander, and Daniel Kifer. "The neural coding framework for
    learning generative models." arXiv preprint arXiv:2012.03405 (2020).

    This model, under the NGC computational framework, is referred to as
    the GNCN-t1-Sigma/Friston, according to the naming convention in
    (Ororbia & Kifer 2022).

    Args:
        args: a Config dictionary containing necessary meta-parameters for the GNCN-t1

    | NOTE:
    | args should contain values for the following:
    | z_top_dim: # of latent variables in layer z3 (top-most layer)
    | z_dim: # of latent variables in layers z1 and z2
    | x_dim: # of latent variables in layer z0 or sensory x
    | seed: number to control determinism of weight initialization
    | wght_sd: standard deviation of Gaussian initialization of weights
    | beta: latent state update factor
    | leak: strength of the leak variable in the latent states
    | K: # of steps to take when conducting iterative inference/settling
    | act_fx: activation function for layers z1, z2, and z3
    | out_fx: activation function for layer mu0 (prediction of z0) (Default: sigmoid)

    @author: Alexander Ororbia
    """
    def __init__(self, args):
        self.args = args

        z_top_dim = int(self.args.getArg("z_top_dim"))
        z_dim = int(self.args.getArg("z_dim"))
        x_dim = int(self.args.getArg("x_dim"))

        seed = int(self.args.getArg("seed")) #69
        beta = float(self.args.getArg("beta"))
        K = int(self.args.getArg("K"))
        act_fx = self.args.getArg("act_fx") #"tanh"
        out_fx = "sigmoid"
        if self.args.hasArg("out_fx") == True:
            out_fx = self.args.getArg("out_fx")
        leak = float(self.args.getArg("leak")) #0.0
        ex_scale = 1.0 #float(self.args.getArg("ex_scale")) #0.0
        n_group = int(self.args.getArg("n_group")) #18
        n_top_group = int(self.args.getArg("n_top_group")) #18
        alpha_scale = float(self.args.getArg("alpha_scale"))
        beta_scale = float(self.args.getArg("beta_scale"))

        use_dfx = False # False --> recovers original GNCN-PDH
        if self.args.hasArg("use_dfx"):
            use_dfx = (self.args.getArg("use_dfx").lower() == 'true')
            print(" > Using Activation Function Derivative...")
        integrate_cfg = {"integrate_type" : "euler", "use_dfx" : use_dfx}
        #lmbda = float(self.args.getArg("lmbda")) #0.0002
        prior_cfg = None #{"prior_type" : "laplace", "lambda" : lmbda}
        precis_cfg = ("uniform", 0.01)
        # alpha_scale = 0.15
        # beta_scale = 0.1
        lat_init = ("lkwta",n_group,alpha_scale,beta_scale)
        lateral_cfg = {"type" : "dense", "has_bias": False, "init" : lat_init, "coeff": -1.0}
        use_mod_factor = False #(self.args.getArg("use_mod_factor").lower() == 'true')
        add_extra_skip =  False #(self.args.getArg("add_extra_skip").lower() == 'true')
        use_skip_error = False # (self.args.getArg("use_skip_error").lower() == 'true')

        z3 = SNode(name="z3", dim=z_top_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
        mu2 = SNode(name="mu2", dim=z_dim, act_fx="relu", zeta=0.0)
        e2 = ENode(name="e2", dim=z_dim, precis_kernel=precis_cfg)
        #e2.use_mod_factor = use_mod_factor
        z2 = SNode(name="z2", dim=z_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
        mu1 = SNode(name="mu1", dim=z_dim, act_fx="relu", zeta=0.0)
        e1 = ENode(name="e1", dim=z_dim, precis_kernel=precis_cfg)
        #e1.use_mod_factor = use_mod_factor
        z1 = SNode(name="z1", dim=z_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
        mu0 = SNode(name="mu0", dim=x_dim, act_fx=out_fx, zeta=0.0)
        e0 = ENode(name="e0", dim=x_dim, ex_scale=ex_scale) #, precis_kernel=precis_cfg)
        z0 = SNode(name="z0", dim=x_dim)

        # create cable wiring scheme relating nodes to one another
        wght_sd = float(self.args.getArg("wght_sd")) #0.025 #0.05 # 0.055
        dcable_cfg = {"type": "dense", "has_bias": False, "init" : ("gaussian",wght_sd), "seed" : seed}
        ecable_cfg = {"type": "dense", "has_bias": False, "init" : ("gaussian",wght_sd), "seed" : seed}
        pos_scable_cfg = {"type": "simple", "coeff": 1.0}
        neg_scable_cfg = {"type": "simple", "coeff": -1.0}

        lat_init_top = ("lkwta",n_top_group,alpha_scale,beta_scale)
        lateral_cfg_top = {"type" : "dense", "has_bias": False, "init" : lat_init_top, "coeff": -1.0}
        z3_to_z3 = z3.wire_to(z3, src_var="phi(z)", dest_var="dz", cable_kernel=lateral_cfg_top) # lateral recurrent connection

        z3_mu2 = z3.wire_to(mu2, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        mu2.wire_to(e2, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
        z2.wire_to(e2, src_var="phi(z)", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
        e2_z3 = e2.wire_to(z3, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)
        e2.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=neg_scable_cfg)

        z2_to_z2 = z2.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=lateral_cfg) # lateral recurrent connection

        z2_mu1 = z2.wire_to(mu1, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        z3_mu1 = z3.wire_to(mu1, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        mu1.wire_to(e1, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
        z1.wire_to(e1, src_var="phi(z)", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
        e1_z2 = e1.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)
        e1.wire_to(z1, src_var="phi(z)", dest_var="dz", cable_kernel=neg_scable_cfg)
        if use_skip_error is True:
            e1_z3 = e1.wire_to(z3, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)

        z1_to_z1 = z1.wire_to(z1, src_var="phi(z)", dest_var="dz", cable_kernel=lateral_cfg) # lateral recurrent connection

        z1_mu0 = z1.wire_to(mu0, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        z2_mu0 = z2.wire_to(mu0, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        if add_extra_skip is True:
            z3_mu0 = z3.wire_to(mu0, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        mu0.wire_to(e0, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
        z0.wire_to(e0, src_var="phi(z)", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
        e0_z1 = e0.wire_to(z1, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)
        e0.wire_to(z0, src_var="phi(z)", dest_var="dz", cable_kernel=neg_scable_cfg)
        if use_skip_error is True:
            e0_z2 = e0.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)
        if add_extra_skip is True:
            if use_skip_error is True:
                e0_z3 = e0.wire_to(z3, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)

        # set up update rules and make relevant edges aware of these
        z3_mu1.set_update_rule(preact=(z3,"phi(z)"), postact=(e1,"phi(z)"), use_mod_factor=use_mod_factor)
        z2_mu0.set_update_rule(preact=(z2,"phi(z)"), postact=(e0,"phi(z)"), use_mod_factor=use_mod_factor)
        if add_extra_skip is True:
            z3_mu0.set_update_rule(preact=(z3,"phi(z)"), postact=(e0,"phi(z)"), use_mod_factor=use_mod_factor)
        z3_mu2.set_update_rule(preact=(z3,"phi(z)"), postact=(e2,"phi(z)"), use_mod_factor=use_mod_factor)
        z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"), use_mod_factor=use_mod_factor)
        z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"), use_mod_factor=use_mod_factor)
        e_gamma = 1.0
        e2_z3.set_update_rule(preact=(e2,"phi(z)"), postact=(z3,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
        e1_z2.set_update_rule(preact=(e1,"phi(z)"), postact=(z2,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
        e0_z1.set_update_rule(preact=(e0,"phi(z)"), postact=(z1,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
        if use_skip_error is True:
            e0_z2.set_update_rule(preact=(e0,"phi(z)"), postact=(z2,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
            e1_z3.set_update_rule(preact=(e1,"phi(z)"), postact=(z3,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
        if add_extra_skip is True:
            if use_skip_error is True:
                e0_z3.set_update_rule(preact=(e0,"phi(z)"), postact=(z3,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)

        # Set up graph - execution cycle/order
        print(" > Constructing NGC graph")
        ngc_model = NGCGraph(K=K)
        ngc_model.proj_update_mag = -1.0 #-1.0
        ngc_model.proj_weight_mag = 1.0
        ngc_model.set_cycle(nodes=[z3,z2,z1,z0])
        ngc_model.set_cycle(nodes=[mu2,mu1,mu0])
        ngc_model.set_cycle(nodes=[e2,e1,e0])
        ngc_model.apply_constraints()
        self.ngc_model = ngc_model

        # build this NGC model's sampling graph
        z3_dim = ngc_model.getNode("z3").dim
        z2_dim = ngc_model.getNode("z2").dim
        z1_dim = ngc_model.getNode("z1").dim
        z0_dim = ngc_model.getNode("z0").dim
        # Set up complementary sampling graph to use in conjunction w/ NGC-graph
        s3 = FNode(name="s3", dim=z3_dim, act_fx=act_fx)
        s2 = FNode(name="s2", dim=z2_dim, act_fx=act_fx)
        s1 = FNode(name="s1", dim=z1_dim, act_fx=act_fx)
        s0 = FNode(name="s0", dim=z0_dim, act_fx=out_fx)
        s3_s2 = s3.wire_to(s2, src_var="phi(z)", dest_var="dz", point_to_path=z3_mu2)
        s2_s1 = s2.wire_to(s1, src_var="phi(z)", dest_var="dz", point_to_path=z2_mu1)
        s3_s1 = s3.wire_to(s1, src_var="phi(z)", dest_var="dz", point_to_path=z3_mu1)
        s1_s0 = s1.wire_to(s0, src_var="phi(z)", dest_var="dz", point_to_path=z1_mu0)
        s2_s0 = s2.wire_to(s0, src_var="phi(z)", dest_var="dz", point_to_path=z2_mu0)
        if add_extra_skip is True:
            s3_s0 = s3.wire_to(s0, src_var="phi(z)", dest_var="dz", point_to_path=z3_mu0)
        sampler = ProjectionGraph()
        sampler.set_cycle(nodes=[s3,s2,s1,s0])
        self.ngc_sampler = sampler

    def project(self, z_sample):
        """
        Run projection scheme to get a sample of the underlying directed
        generative model given the clamped variable *z_sample*

        Args:
            z_sample: the input noise sample to project through the NGC graph

        Returns:
            x_sample (sample(s) of the underlying generative model)
        """
        readouts = self.ngc_sampler.project(
                        clamped_vars=[("s3",tf.cast(z_sample,dtype=tf.float32))],
                        readout_vars=[("s0","phi(z)")]
                    )
        x_sample = readouts[0][2]
        return x_sample

    def settle(self, x):
        """
        Run an iterative settling process to find latent states given clamped
        input and output variables

        Args:
            x: sensory input to reconstruct/predict

        Returns:
            x_hat (predicted x)
        """
        readouts = self.ngc_model.settle(
                        clamped_vars=[("z0", x)],
                        readout_vars=[("mu0","phi(z)"),("mu1","phi(z)"),("mu2","phi(z)")]
                    )
        x_hat = readouts[0][2]
        return x_hat

    def calc_updates(self, avg_update=True):
        """
        Calculate adjustments to parameters under this given model and its
        current internal state values

        Returns:
            delta, a list of synaptic matrix updates (that follow order of .theta)
        """
        Ns = self.ngc_model.extract("z0","phi(z)").shape[0]
        delta = self.ngc_model.calc_updates()
        if avg_update is True:
            for p in range(len(delta)):
                delta[p] = delta[p] * (1.0/(Ns * 1.0))
        return delta

    def update(self, x, avg_update=True):
        """
        Updates synaptic parameters/connections given sensory input

        Args:
            x: a sensory sample or batch of sensory samples
        """
        self.settle(x)
        delta = self.calc_updates(avg_update=avg_update)
        self.opt.apply_gradients(zip(delta, self.ngc_model.theta))
        self.ngc_model.apply_constraints()

    def clear(self):
        """Clears the states/values of the stateful nodes in this NGC system"""
        self.ngc_model.clear()
        self.ngc_sampler.clear()

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
        into this model's weight variables
        """
        #self.param_var = copy.deepcopy(source.param_var)
        if tau >= 0.0:
            for l in range(0, len(self.ngc_model.theta)):
                self.ngc_model.theta[l].assign( self.ngc_model.theta[l] * (1 - tau) + source.ngc_model.theta[l] * tau )
        else:
            for l in range(0, len(self.ngc_model.theta)):
                self.ngc_model.theta[l].assign( source.ngc_model.theta[l] )
