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

class GNCN_t1_SC:
    """
    Structure for constructing the sparse coding model proposed in:

    Olshausen, B., Field, D. Emergence of simple-cell receptive field properties
    by learning a sparse code for natural images. Nature 381, 607–609 (1996).

    Note this model imposes a factorial (Cauchy) prior to induce sparsity in the latent
    activities z1 (the latent codebook). Synapses initialized from a (fan-in) scaled
    uniform distribution.
    This model would be named, under the NGC computational framework naming convention
    (Ororbia & Kifer 2022), as the GNCN-t1/SC (SC = sparse coding) or GNCN-t1/Olshausen.

    | Node Name Structure:
    | p(z1) ; z1 -(z1-mu0)-> mu0 ;e0; z0
    | Cauchy prior applied for p(z1)

    Note: You can also recover the model learned through ISTA by using, instead of
    a factorial prior over latents, a thresholding function such as the
    "soft_threshold". (Make sure you set "prior" to "none" in this case.)
    This results in the GNCN-t1/SC emulating a system similar
    to that proposed in:

    Daubechies, Ingrid, Michel Defrise, and Christine De Mol. "An iterative
    thresholding algorithm for linear inverse problems with a sparsity constraint."
    Communications on Pure and Applied Mathematics: A Journal Issued by the
    Courant Institute of Mathematical Sciences 57.11 (2004): 1413-1457.

    Args:
        args: a Config dictionary containing necessary meta-parameters for the GNCN-t1/SC

    | DEFINITION NOTE:
    | args should contain values for the following:
    | * batch_size - the fixed batch-size to be fed into this model
    | * z_dim - # of latent variables in layers z1
    | * x_dim - # of latent variables in layer z0 or sensory x
    | * seed - number to control determinism of weight initialization
    | * beta - latent state update factor
    | * leak - strength of the leak variable in the latent states (Default = 0)
    | * prior - type of prior to use (Default = "cauchy")
    | * lmbda - strength of the prior applied over latent state activities (only if prior != "none")
    | * threshold - type of threshold to use (Default = "none")
    | * thr_lmbda - strength of the threshold applied over latent state activities (only if threshold != "none")
    | * n_group - must be > 0 if lat_type != None and s.t. (z_dim mod n_group) == 0
    | * K - # of steps to take when conducting iterative inference/settling
    | * act_fx - activation function for layers z1 (Default = identity)
    | * out_fx - activation function for layer mu0 (prediction of z0) (Default: identity)

    """
    def __init__(self, args):
        self.args = args

        batch_size = int(self.args.getArg("batch_size"))
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
        prior = "cauchy"
        lmbda = 0.0
        prior_cfg = None
        if self.args.hasArg("prior") == True:
            prior = self.args.getArg("prior")
            lmbda = float(self.args.getArg("lambda"))
            prior_cfg = {"prior_type" : prior, "lambda" : lmbda}
        threshold = "none"
        thr_lbmda = 0.0
        thr_cfg = None
        if self.args.hasArg("threshold") == True:
            threshold = self.args.getArg("threshold")
            thr_lbmda = float(self.args.getArg("thr_lambda"))
            thr_cfg = {"threshold_type" : threshold, "thr_lambda" : thr_lbmda}

        # set up state integration function
        integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}
        use_mod_factor = False #(self.args.getArg("use_mod_factor").lower() == 'true')

        # set up system nodes
        z1 = SNode(name="z1", dim=z_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg, prior_kernel=prior_cfg,
                   threshold_kernel=thr_cfg)
        mu0 = SNode(name="mu0", dim=x_dim, act_fx=out_fx, zeta=0.0)
        e0 = ENode(name="e0", dim=x_dim)
        z0 = SNode(name="z0", dim=x_dim, beta=beta, integrate_kernel=integrate_cfg, leak=0.0)

        # create cable wiring scheme relating nodes to one another
        init_kernels = {"A_init" : ("unif_scale",1.0)}
        dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : seed}
        pos_scable_cfg = {"type": "simple", "coeff": 1.0}
        neg_scable_cfg = {"type": "simple", "coeff": -1.0}
        param_axis = 1
        constraint_cfg = {"clip_type":"forced_norm_clip","clip_mag":1.0,"clip_axis":param_axis}

        z1_mu0 = z1.wire_to(mu0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg,
                            short_name="W1")
        z1_mu0.set_constraint(constraint_cfg)
        mu0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg,
                    short_name="1")
        z0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg,
                   short_name="1")
        e0.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z1_mu0,"A^T"),
                   short_name="W1^T")
        e0.wire_to(z0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg,
                   short_name="-1")
        z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"), param=["A"])
        #param_axis = 1

        #z1_to_z1 = z1.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(z1_mu0,"A*A^T"))

        ### alternative way to construct the same model above ###
        #### this wiring pattern below means the feedback/error weights are leanred
        #### while the forward/generative weights are tied to them
        #e0_z1 = e0.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", cable_kernel=dcable_cfg)
        #z1_mu0 = z1.wire_to(mu0, src_comp="phi(z)", dest_comp="dz_td", mirror_path_kernel=(e0_z1,"A^T"))
        #mu0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=pos_scable_cfg)
        #z0.wire_to(e0, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=pos_scable_cfg)
        #e0.wire_to(z0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=neg_scable_cfg)
        #e0_z1.set_update_rule(preact=(e0,"phi(z)"), postact=(z1,"phi(z)")) # update rule
        # param_axis = 0

        # Set up graph - execution cycle/order
        print(" > Constructing NGC graph")
        ngc_model = NGCGraph(K=K, name="gncn_t1_sc", batch_size=batch_size)
        ngc_model.set_cycle(nodes=[z1, z0])
        ngc_model.set_cycle(nodes=[mu0])
        ngc_model.set_cycle(nodes=[e0])
        info = ngc_model.compile()
        self.info = parse_simulation_info(info)
        #ngc_model.apply_constraints()
        self.ngc_model = ngc_model

        # build this NGC model's sampling graph
        z1_dim = ngc_model.getNode("z1").dim
        z0_dim = ngc_model.getNode("z0").dim
        # Set up complementary sampling graph to use in conjunction w/ NGC-graph
        s1 = FNode(name="s1", dim=z1_dim, act_fx=act_fx)
        s0 = FNode(name="s0", dim=z0_dim, act_fx=out_fx)
        s1_s0 = s1.wire_to(s0, src_comp="phi(z)", dest_comp="dz", mirror_path_kernel=(z1_mu0,"A"))
        sampler = ProjectionGraph()
        sampler.set_cycle(nodes=[s1, s0])
        sampler_info = sampler.compile()
        self.sampler_info = parse_simulation_info(sampler_info)
        self.ngc_sampler = sampler

        self.delta = None

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
                        clamped_vars=[("s1","z",tf.cast(z_sample,dtype=tf.float32))],
                        readout_vars=[("s0","phi(z)")]
                    )
        x_sample = readouts[0][2]
        return x_sample

    def settle(self, x, K=-1, cold_start=True, calc_update=True):
        """
        Run an iterative settling process to find latent states given clamped
        input and output variables

        Args:
            x: sensory input to reconstruct/predict

            K: number of steps to run iterative settling for

            cold_start: start settling process states from zero (Leave this to True)

            calc_update: if True, computes synaptic updates @ end of settling
                process (Default = True)

        Returns:
            x_hat (predicted x)
        """
        K_ = K
        if K_ <= 0:
            K_ = self.ngc_model.K
        readouts, delta = self.ngc_model.settle(
                        clamped_vars=[("z0","z",x)],
                        readout_vars=[("mu0","phi(z)")],
                        K=K_,
                        cold_start=cold_start,
                        calc_delta=calc_update
                    )
        self.delta = delta # store delta to constructor for later retrieval
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
        #delta = self.ngc_model.calc_updates()
        delta = self.delta
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
        _, delta = self.settle(x, y, calc_update=True)
        self.delta = delta
        self.opt.apply_gradients(zip(delta, self.ngc_model.theta))
        self.ngc_model.apply_constraints()

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
