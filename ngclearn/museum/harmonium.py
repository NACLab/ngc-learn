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

class Harmonium:
    """
    Structure for constructing the sparse coding model proposed in:

    Hinton, Geoffrey E. "Training products of experts by maximizing contrastive
    likelihood." Technical Report, Gatsby computational neuroscience unit (1999).

    | Node Name Structure:
    | z1 -(z1-z0)-> z0
    | z0 -(z0-z1)-> z1
    | Note: z1-z0 = (z0-z1)^T (transpose-tied synapses)

    Args:
        args: a Config dictionary containing necessary meta-parameters for the harmonium

    | DEFINITION NOTE:
    | args should contain values for the following:
    | * batch_size - the fixed batch-size to be fed into this model
    | * z_dim - # of latent variables in layers z1
    | * x_dim - # of latent variables in layer z0 (or sensory x)
    | * seed - number to control determinism of weight initialization
    | * wght_sd - standard deviation of Gaussian initialization of weights
    | * K - # of steps to take when conducting Contrastive Divergence
    | * act_fx - activation function for layers z1 (Default: sigmoid)
    | * out_fx - activation function for layer z0 (prediction of z0) (Default: sigmoid)

    """
    def __init__(self, args):
        self.args = args

        batch_size = int(self.args.getArg("batch_size"))
        z_dim = int(self.args.getArg("z_dim"))
        x_dim = int(self.args.getArg("x_dim"))

        seed = int(self.args.getArg("seed"))
        K = int(self.args.getArg("K"))
        act_fx = self.args.getArg("act_fx") #"tanh"
        out_fx = "identity"
        if self.args.hasArg("out_fx") == True:
            out_fx = self.args.getArg("out_fx")
        wght_sd = float(self.args.getArg("wght_sd"))

        # set up state integration function
        integrate_cfg = {"integrate_type" : "euler", "use_dfx" : False}
        init_kernels = {"A_init" : ("gaussian",wght_sd), "b_init" : ("zeros")}
        dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : seed}
        pos_scable_cfg = {"type": "simple", "coeff": 1.0}
        neg_scable_cfg = {"type": "simple", "coeff": -1.0}
        #constraint_cfg = {"clip_type":"forced_norm_clip","clip_mag":1.0,"clip_axis":1}

        # set up positive phase nodes
        z1 = SNode(name="z1", dim=z_dim, beta=1, act_fx=act_fx, zeta=0.0,
                     integrate_kernel=integrate_cfg)
        z0 = SNode(name="z0", dim=x_dim, beta=1, act_fx="identity", zeta=0.0,
                     integrate_kernel=integrate_cfg)
        z0_z1 = z0.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", cable_kernel=dcable_cfg)
        z1_z0 = z1.wire_to(z0, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z0_z1,"A^T"),
                           cable_kernel=dcable_cfg)
        #z0_z1.set_constraint(constraint_cfg)

        # set up positive phase update
        z0_z1.set_update_rule(preact=(z0,"phi(z)"), postact=(z1,"phi(z)"), param=["A","b"])
        z1_z0.set_update_rule(postact=(z0,"phi(z)"), param=["b"])

        # build positive graph
        print(" > Constructing Positive Phase Graph")
        pos_phase = NGCGraph(K=1, name="rbm_pos")
        pos_phase.set_cycle(nodes=[z0, z1]) # z0 -> z1
        pos_phase.apply_constraints()
        pos_phase.set_learning_order([z1_z0, z0_z1]) 
        info = pos_phase.compile(batch_size=batch_size, use_graph_optim=True)
        self.pos_info = parse_simulation_info(info)
        self.pos_phase = pos_phase

        # set up negative phase nodes
        z1n_i = SNode(name="z1n_i", dim=z_dim, beta=1, act_fx=act_fx, zeta=0.0,
                     integrate_kernel=integrate_cfg)
        z0n = SNode(name="z0n", dim=x_dim, beta=1, act_fx=out_fx, zeta=0.0,
                     integrate_kernel=integrate_cfg)
        z1n = SNode(name="z1n", dim=z_dim, beta=1, act_fx=act_fx, zeta=0.0,
                     integrate_kernel=integrate_cfg)
        n1_n0 = z1n_i.wire_to(z0n, src_comp="phi(z)", dest_comp="dz_td", mirror_path_kernel=(z0_z1,"A^T"),
                            cable_kernel=dcable_cfg) # reuse A but create new b
        n0_n1 = z0n.wire_to(z1n, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z0_z1,"A+b")) # reuse A  & b

        # set up negative phase update
        n0_n1.set_update_rule(preact=(z0n,"phi(z)"), postact=(z1n,"phi(z)"), param=["A","b"])
        n1_n0.set_update_rule(postact=(z0n,"phi(z)"), param=["b"])

        # build negative graph
        print(" > Constructing Negative Phase Graph")
        neg_phase = NGCGraph(K=K, name="rbm_neg")
        neg_phase.set_cycle(nodes=[z1n_i, z0n, z1n]) # z1 -> z0 -> z1
        neg_phase.apply_constraints()
        neg_phase.set_learning_order([n1_n0, n0_n1]) # forces order: c, W, b
        info = neg_phase.compile(batch_size=batch_size, use_graph_optim=False)
        self.neg_info = parse_simulation_info(info)
        self.neg_phase = neg_phase

        self.theta = self.pos_phase.theta
        self.pos_delta = None
        self.neg_delta = None

    def settle(self, x, calc_update=True):
        """
        Run an iterative settling process to find latent states given clamped
        input and output variables.

        Args:
            x: sensory input to reconstruct/predict

            calc_update: if True, computes synaptic updates @ end of settling
                process for both NGC system and inference co-model (Default = True)

        Returns:
            x_hat (predicted x)
        """
        ## run positive phase
        readouts, delta = self.pos_phase.settle(
                            clamped_vars=[("z0","z", x)],
                            readout_vars=[("z1","phi(z)")],
                            calc_delta=calc_update
                          )
        self.pos_delta = delta

        z1_pos = readouts[0][2]
        ## run negative phase
        readouts, delta = self.neg_phase.settle(
                            init_vars=[("z1n_i","phi(z)", z1_pos)],
                            readout_vars=[("z0n","phi(z)"),("z1n","z")],
                            calc_delta=calc_update
                          )
        self.neg_delta = delta
        ## return reconstruction (from negative phase)
        x_hat = readouts[0][2]
        return x_hat

    def calc_updates(self, avg_update=True, decay_rate=-1.0): # decay_rate=0.001
        """
        Calculate adjustments to parameters under this given model and its
        current internal state values

        Returns:
            delta, a list of synaptic updates (that follow order of pos_phase.theta)
        """
        Ns = self.pos_phase.extract("z0","phi(z)").shape[0]
        delta = []
        for i in range(len(self.pos_delta)):
            pos_dx = self.pos_delta[i]
            neg_dx = self.neg_delta[i]
            dx = ( pos_dx - neg_dx )
            delta.append(dx) # multiply CD update by -1 to allow for minimization

        if avg_update is True:
            for p in range(len(delta)):
                delta[p] = delta[p] * (1.0/(Ns * 1.0))
                if decay_rate > 0.0: # weight decay
                    delta[p] = delta[p] - (self.pos_phase.theta[p] * decay_rate)
        return delta

    def clear(self):
        """Clears the states/values of the stateful nodes in this NGC system"""
        self.pos_phase.clear()
        self.neg_phase.clear()
        self.pos_delta = None
        self.neg_delta = None

    def print_norms(self):
        """Prints the Frobenius norms of each parameter of this system"""
        str = ""
        for param in self.pos_phase.theta:
            str = "{} | {} : {}".format(str, param.name, tf.norm(param,ord=2))
        #str = "{}\n".format(str)
        return str
