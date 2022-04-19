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

"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU LGPL-3.0-or-later license.

You should have received a copy of the XYZ license with
this file. If not, please write to: ago@cs.rit.edu , or visit:
https://www.gnu.org/licenses/lgpl-3.0.en.html
"""

class GNCN_t2_FFN:
    """
    Structure for constructing a feedforward-like, x->y NGC circuit from the
    paper Ororbia & Mali 2021, AAAI.<br>
    According to the naming convention of Ororbia & Kifer 2021, this model's
    name is the GNCN-t2.

    @author: Alex Ororbia
    """
    def __init__(self, in_dim, out_dim, z_dim=128, act_fx = "relu", out_fx="identity",
                 K=10, beta=0.1, leak=0.0001, seed=69, wght_sd=0.025, lat_dims=None,
                 has_bias=False, use_mod_factor=False, e_gamma=0.0, lmbda=0.0001):
        # set integration method for evolving units over time
        use_dfx = True
        integrate_cfg = {"integrate_type" : "euler", "use_dfx" : use_dfx}

        # create cable wiring scheme relating nodes to one another
        #wght_sd = 0.025
        dcable_cfg = {"type": "dense", "has_bias": has_bias, "init" : ("gaussian",wght_sd), "seed" : seed}
        ecable_cfg = {"type": "dense", "has_bias": False, "init" : ("gaussian",wght_sd), "seed" : seed}
        pos_scable_cfg = {"type": "simple", "coeff": 1.0}
        neg_scable_cfg = {"type": "simple", "coeff": -1.0}

        prior_cfg = None
        if lmbda > 0.0:
            prior_cfg = {"prior_type" : "laplace", "lambda" : lmbda}
        # n_group = 18
        # alpha_scale = 0.15
        # beta_scale = 0.1
        # lat_init = ("lkwta",n_group,alpha_scale,beta_scale)
        # lateral_cfg = {"type" : "dense", "has_bias": False, "init" : lat_init, "coeff": -1.0}

        z2_dim = z_dim
        z1_dim = z_dim
        if lat_dims is not None:
            z2_dim = lat_dims[0]
            z1_dim = lat_dims[1]

        # input layer == top-most layer in this NGC structure
        g_fx = "identity" #"identity" # relu
        z3 = SNode(name="z3", dim=in_dim, beta=beta, leak=leak, act_fx="identity",
                   integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
        mu2 = SNode(name="mu2", dim=z2_dim, act_fx=g_fx, zeta=0.0)
        e2 = ENode(name="e2", dim=z2_dim, precis_kernel=None)
        # latent layer 2
        z2 = SNode(name="z2", dim=z2_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
        mu1 = SNode(name="mu1", dim=z1_dim, act_fx=g_fx, zeta=0.0)
        e1 = ENode(name="e1", dim=z1_dim, precis_kernel=None)
        # latent layer 1
        z1 = SNode(name="z1", dim=z1_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
        mu0 = SNode(name="mu0", dim=out_dim, act_fx=out_fx, zeta=0.0)
        e0 = ENode(name="e0", dim=out_dim, ex_scale=1.0, precis_kernel=None)
        # output layer
        z0 = SNode(name="z0", dim=out_dim)

        # dscale = 1 #-1.0
        # e2_tm1 = SNode(name="e2_tm1", dim=z2_dim, beta=1.0, leak=0.0, act_fx="identity", zeta=0.0)
        # e1_tm1 = SNode(name="e1_tm1", dim=z1_dim, beta=1.0, leak=0.0, act_fx="identity", zeta=0.0)
        # de2 = SNode(name="de2", dim=z2_dim, beta=dscale, leak=0.0, act_fx="identity", zeta=0.0)
        # de1 = SNode(name="de1", dim=z1_dim, beta=dscale, leak=0.0, act_fx="identity", zeta=0.0)

        z3_mu2 = z3.wire_to(mu2, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        mu2.wire_to(e2, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
        z2.wire_to(e2, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
        #e2_z3 = e2.wire_to(z3, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)
        e2.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=neg_scable_cfg)

        #z2_to_z2 = z2.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=lateral_cfg) # lateral recurrent connection

        z2_mu1 = z2.wire_to(mu1, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        mu1.wire_to(e1, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
        z1.wire_to(e1, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
        e1_z2 = e1.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)
        e1.wire_to(z1, src_var="phi(z)", dest_var="dz", cable_kernel=neg_scable_cfg)

        #z2_mu0 = z2.wire_to(mu0, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)

        #z1_to_z1 = z1.wire_to(z1, src_var="phi(z)", dest_var="dz", cable_kernel=lateral_cfg) # lateral recurrent connection

        z1_mu0 = z1.wire_to(mu0, src_var="phi(z)", dest_var="dz", cable_kernel=dcable_cfg)
        mu0.wire_to(e0, src_var="phi(z)", dest_var="pred_mu", cable_kernel=pos_scable_cfg)
        z0.wire_to(e0, src_var="z", dest_var="pred_targ", cable_kernel=pos_scable_cfg)
        e0_z1 = e0.wire_to(z1, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)
        e0.wire_to(z0, src_var="phi(z)", dest_var="dz", cable_kernel=neg_scable_cfg)


        # z2.wire_to(e2_tm1, src_var="z", dest_var="dz", cable_kernel=pos_scable_cfg)
        # z2.wire_to(de2, src_var="z", dest_var="dz", cable_kernel=pos_scable_cfg)
        # e2_tm1.wire_to(de2, src_var="z", dest_var="dz", cable_kernel=neg_scable_cfg)
        #
        # z1.wire_to(e1_tm1, src_var="z", dest_var="dz", cable_kernel=pos_scable_cfg)
        # z1.wire_to(de1, src_var="z", dest_var="dz", cable_kernel=pos_scable_cfg)
        # e1_tm1.wire_to(de1, src_var="z", dest_var="dz", cable_kernel=neg_scable_cfg)

        #e0_z2 = e0.wire_to(z2, src_var="phi(z)", dest_var="dz", cable_kernel=ecable_cfg)

        ################################################################################
        # set up update rules and make relevant edges aware of these
        ################################################################################
        z3_mu2.set_update_rule(preact=(z3,"phi(z)"), postact=(e2,"phi(z)"), use_mod_factor=use_mod_factor)
        z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"), use_mod_factor=use_mod_factor)
        z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"), use_mod_factor=use_mod_factor)
        #e_gamma = 0. #0.0001

        if e_gamma > 0.0:
            #e2_z3.set_update_rule(preact=(e2,"phi(z)"), postact=(z3,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
            #e1_z2.set_update_rule(preact=(e1,"phi(z)"), postact=(z2,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
            #e0_z1.set_update_rule(preact=(e0,"phi(z)"), postact=(z1,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)

            e1_z2.set_update_rule(preact=(e1,"phi(z)"), postact=(e2,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)
            e0_z1.set_update_rule(preact=(e0,"phi(z)"), postact=(e1,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)

            #e0_z2.set_update_rule(preact=(e0,"phi(z)"), postact=(z2,"phi(z)"), gamma=e_gamma, use_mod_factor=use_mod_factor)

        ################################################################################
        # Set up graph - execution cycle/order
        ################################################################################
        print(" > Constructing NGC graph")
        agent = NGCGraph(K=K)
        agent.proj_update_mag = -1.0 #-1.0 #-1.0
        agent.proj_weight_mag = -1.0 #2.0 #2.0 #1.0
        #agent.set_cycle(nodes=[e2_tm1,e1_tm1]) # exec state calc
        agent.set_cycle(nodes=[z3,z2,z1,z0]) # exec state calc
        agent.set_cycle(nodes=[mu2,mu1,mu0]) # exec expectation calc
        agent.set_cycle(nodes=[e2,e1,e0]) # exec error unit calc
        #agent.set_cycle(nodes=[de2,de1]) # exec state calc
        #agent.apply_constraints()
        if e_gamma <= 0.0:
            agent.set_evolve_pair( ("e0-to-z1_dense", "z1-to-mu0_dense" ) )
            agent.set_evolve_pair( ("e1-to-z2_dense", "z2-to-mu1_dense" ) )

        self.ngc_model = agent

        # Set up complementary sampling graph to use in conjunction w/ NGC-graph
        z3_dim = agent.getNode("z3").dim
        z2_dim = agent.getNode("z2").dim
        z1_dim = agent.getNode("z1").dim
        z0_dim = agent.getNode("z0").dim
        s3 = FNode(name="s3", dim=z3_dim, act_fx="identity")
        s2 = FNode(name="s2", dim=z2_dim, act_fx=act_fx)
        s1 = FNode(name="s1", dim=z1_dim, act_fx=act_fx)
        s0 = FNode(name="s0", dim=z0_dim, act_fx=out_fx)
        s3_s2 = s3.wire_to(s2, src_var="phi(z)", dest_var="dz", point_to_path=z3_mu2, cable_kernel=dcable_cfg)
        s2_s1 = s2.wire_to(s1, src_var="phi(z)", dest_var="dz", point_to_path=z2_mu1, cable_kernel=dcable_cfg)
        s1_s0 = s1.wire_to(s0, src_var="phi(z)", dest_var="dz", point_to_path=z1_mu0, cable_kernel=dcable_cfg)
        sampler = ProjectionGraph()
        sampler.set_cycle(nodes=[s3,s2,s1,s0])

        self.ngc_sampler = sampler

        eta_v = 0.002 #0.001
        self.opt = tf.keras.optimizers.Adam(eta_v)
        #eta_v = 0.1
        #self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=eta_v)

    def print_norms(self):
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

    def predict(self, input): # wrapper function
        return self.project(input)

    def project(self, input):
        """
        Run projection scheme to get a fast prediction given
        a clamped input variable (Ororbia & Kifer 2021)
        """
        #tf.cast(input,dtype=tf.float32)
        readouts = self.ngc_sampler.project(clamped_vars=[("s3",input)],
                                            readout_vars=[("s0","phi(z)")])
        #output = readouts[0][2]
        output = self.ngc_sampler.extract("s0","phi(z)")
        return output

    def settle(self, input, target, project_init=False, debug=False):
        """
        Run iterative settling process to find latent states given clamped
        input and output variables
        """
        if project_init is True:
            # run projection
            self.ngc_sampler.project(clamped_vars=[("s3",input)])
            #z_tmp = self.ngc_sampler.extract("s3","z")
            #self.ngc_model.clamp("z3", z_tmp, is_persistent=False)
            z_tmp = self.ngc_sampler.extract("s2","z")
            self.ngc_model.clamp(node_name="z2", data=("z",z_tmp+0), is_persistent=False)
            z_tmp = self.ngc_sampler.extract("s2","phi(z)")
            self.ngc_model.clamp(node_name="z2", data=("phi(z)",z_tmp+0), is_persistent=False)
            z_tmp = self.ngc_sampler.extract("s1","z")
            self.ngc_model.clamp(node_name="z1", data=("z",z_tmp+0), is_persistent=False)
            z_tmp = self.ngc_sampler.extract("s1","phi(z)")
            self.ngc_model.clamp(node_name="z1", data=("phi(z)",z_tmp+0), is_persistent=False)
            #e0 = target - self.ngc_sampler.extract("s0","phi(z)")
            #self.ngc_model.clamp(node_name="e0", data=("z",e0+0), is_persistent=False)
            #self.ngc_model.clamp(node_name="e0", data=("phi(z)",e0+0), is_persistent=False)
            self.ngc_sampler.clear()

        # run settling process
        if target is not None:
            readouts = self.ngc_model.settle(clamped_vars=[("z0",target),("z3",input)],
                                             readout_vars=[("mu0","phi(z)")], debug=debug)#,("mu1","phi(z)"),("mu2","phi(z)")])
        else:
            readouts = self.ngc_model.settle(clamped_vars=[("z3",input)],
                                             readout_vars=[("mu0","phi(z)")], debug=debug)#,("mu1","phi(z)"),("mu2","phi(z)")])

        #output = readouts[0][2]
        output = self.ngc_model.extract("mu0","phi(z)")
        #delta = agent.calc_updates()
        return output

    def calc_updates(self, x, avg_update=True):
        """
        Calculate adjustments to parameters under this given model
        """
        delta = self.ngc_model.calc_updates()
        if avg_update is True:
            for p in range(len(delta)):
                delta[p] = delta[p] * (1.0/(x.shape[0] * 1.0))
        return delta

    def calc_err_weight_updates(self, gamma_et=1, decay_et=0.0, rule_type="temp_diff", lambda_e=0.0):
        return self.ngc_model.calc_evolved_cable_updates(gamma_et=gamma_et, decay_et=decay_et,
                                                         rule_type=rule_type, lambda_e=lambda_e)

    def update(self, x, avg_update=False):
        """
        Update synaptic parameters/connections given current state of this model
        """
        delta = self.calc_updates(x, avg_update=avg_update)
        self.opt.apply_gradients(zip(delta, self.ngc_model.theta))
        self.ngc_model.apply_constraints()

    def clear(self):
        self.ngc_model.clear()
        self.ngc_sampler.clear()
