import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils

class SNode(Node):
    """
    | Implements a (rate-coded) state node that follows NGC settling dynamics according to:
    |   d.z/d.t = -z * leak + dz + prior(z), where dz = dz_td + dz_bu * phi'(z)
    | where:
    |   dz - aggregated input signals from other nodes/locations
    |   leak - controls strength of leak variable/decay
    |   prior(z) - distributional prior placed over z (such as a kurtotic prior)

    | Note that the above is used to adjust neural activity values via an integator inside a node.
        For example, if the standard/default Euler integrator is used then the neurons inside this
        node are adjusted per step as follows:
    |   z <- z * zeta + d.z/d.t * beta
    | where:
    |   beta - strength of update to node state z
    |   zeta - controls the strength of recurrent carry-over, if set to 0 no carry-over is used (stateless)

    | Compartments:
    |   * dz_td - the top-down pressure compartment (deposited signals summed)
    |   * dz_bu - the bottom-up pressure compartment, potentially weighted by phi'(x)) (deposited signals summed)
    |   * z - the state neural activities
    |   * phi(z) - the post-activation of the state activities
    |   * S(z) - the sampled state of phi(z) (Default = identity or f(phi(z)) = phi(z))
    |   * mask - a binary mask to be applied to the neural activities

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        beta: strength of update to adjust neurons at each simulation step (Default = 1)

        leak: strength of the leak applied to each neuron (Default = 0)

        zeta: effect of recurrent/stateful carry-over (Defaul = 1)

        act_fx: activation function -- phi(v) -- to apply to neural activities

            :Note: if using either "kwta" or "bkwta", please input how many winners
                should win the competiton, i.e., use "kwta(N)" or "bkwta(N)" where
                N is an integer > 0.

        batch_size: batch-size this node should assume (for use with static graph optimization)

        integrate_kernel: Dict defining the neural state integration process type. The expected keys and
            corresponding value types are specified below:

            :`'integrate_type'`: type integration method to apply to neural activity over time.
                If "euler" is specified, Euler integration will be used (future ngc-learn versions will support
                "midpoint"/other methods).

            :`'use_dfx'`: a boolean that decides if phi'(v) (activation derivative) is used in the integration
                process/update.

            :Note: specifying None will automatically set this node to use Euler integration w/ use_dfx=False

        prior_kernel: Dict defining the type of prior function to apply over neural activities.
            The expected keys and corresponding value types are specified below:

            :`'prior_type'`: type of (centered) distribution to use as a prior over neural activities.
                If "laplace" is specified, a Laplacian distribution is used,
                if "cauchy" is specified, a Cauchy distribution will be used,
                if "gaussian" is specified, a Gaussian distribution will be used, and
                if "exp" is specified, the exponential distribution will be used.

            :`'lambda'`: the scale factor controlling the strength of the prior applied to neural activities.

            :Note: specifying None will result in no prior distribution being applied

        threshold_kernel: Dict defining the type of threshold function to apply over neural activities.
            The expected keys and corresponding value types are specified below:

            :`'threshold_type'`: type of (centered) distribution to use as a prior over neural activities.
                If "soft_threshold" is specified, a soft thresholding function is used, and
                if "cauchy_threshold" is specified, a cauchy thresholding function is used,

            :`'thr_lambda'`: the scale factor controlling the strength of the threshold applied to neural activities.

            :Note: specifying None will result in no threshold function being applied

        trace_kernel: <unused> (Default = None)

        samp_fx: the sampling/stochastic activation function -- S(v) -- to apply
            to neural activities (Default = identity)
    """
    def __init__(self, name, dim, beta=1.0, leak=0.0, zeta=1.0, act_fx="identity",
                 batch_size=1, integrate_kernel=None, prior_kernel=None,
                 threshold_kernel=None, trace_kernel=None, samp_fx="identity"):
        node_type = "state"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.is_clamped = False

        self.beta = beta
        self.leak = leak
        self.zeta = zeta

        self.integrate_kernel = integrate_kernel
        self.prior_kernel = prior_kernel
        self.threshold_kernel = threshold_kernel
        self.use_dfx = False
        self.integrate_type = "euler" # Default = euler
        if integrate_kernel is not None:
            self.use_dfx = integrate_kernel.get("use_dfx")
            self.integrate_type = integrate_kernel.get("integrate_type")
        self.prior_type = None
        self.lmbda = 0.0
        if prior_kernel is not None:
            self.prior_type = prior_kernel.get("prior_type")
            self.lmbda = prior_kernel.get("lambda")
        self.threshold_type = None
        self.thr_lmbda = 0.0
        if threshold_kernel is not None:
            self.threshold_type = threshold_kernel.get("threshold_type")
            self.thr_lmbda = threshold_kernel.get("thr_lambda")

        # build post-activation function
        self.act_fx = act_fx
        fx, dfx = transform_utils.decide_fun(act_fx)
        self.fx = fx
        self.dfx = dfx
        self.n_winners = -1
        if "bkwta" in act_fx:
            self.n_winners = int(act_fx[act_fx.index("(")+1:act_fx.rindex(")")])
        # build stochastic sampling function
        self.samp_fx = samp_fx
        sfx, sdfx = transform_utils.decide_fun(samp_fx)
        self.sfx = sfx
        self.sdfx = sdfx

        self.constant_names = ["beta", "leak", "zeta"]
        self.constants = {}
        self.constants["beta"] = self.beta
        self.constants["leak"] = self.leak
        self.constants["zeta"] = self.zeta

        self.compartment_names = ["dz_bu", "dz_td", "z", "phi(z)", "S(z)"]
        self.compartments = {}
        for name in self.compartment_names:
            if "phi(z)" in name:
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name="{}_phi_z".format(self.name))
            elif "S(z)" in name:
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name="{}_S_z".format(self.name))
            else:
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name="{}_{}".format(self.name, name))
        self.mask_names = ["mask"]
        self.masks = {}
        for name in self.mask_names:
            self.masks[name] = tf.Variable(tf.ones([batch_size,dim]), name="{}_{}".format(self.name, name))

        self.connected_cables = []

    def compile(self):
        info = super().compile()
        info["beta"] = self.beta
        info["leak"] = self.leak
        info["zeta"] = self.zeta
        info["phi(x)"] = self.act_fx
        info["S(x)"] = self.samp_fx
        info["integration.form"] = self.integrate_kernel
        if self.prior_kernel is not None:
            info["prior.form"] = self.prior_kernel
        if self.threshold_kernel is not None:
            info["threshold.form"] = self.threshold_kernel
        return info

    def step(self, injection_table=None, skip_core_calc=False):
        if injection_table is None:
            injection_table = {}

        bmask = self.masks.get("mask")
        ########################################################################
        if skip_core_calc == False:
            # clear any relevant compartments that are NOT stateful before accruing
            # new deposits (this is crucial to ensure any desired stateless properties)
            if self.do_inplace == True:
                if injection_table.get("dz_bu") is None:
                    self.compartments["dz_bu"].assign(self.compartments["dz_bu"] * 0)
                if injection_table.get("dz_td") is None:
                    self.compartments["dz_td"].assign(self.compartments["dz_td"] * 0)
            else:
                if injection_table.get("dz_bu") is None:
                    self.compartments["dz_bu"] = (self.compartments["dz_bu"] * 0)
                if injection_table.get("dz_td") is None:
                    self.compartments["dz_td"] = (self.compartments["dz_td"] * 0)

            # gather deposits from any connected nodes & insert them into the
            # right compartments/regions -- deposits in this logic are linearly combined
            for cable in self.connected_cables:
                deposit = cable.propagate()
                dest_comp = cable.dest_comp
                if injection_table.get(dest_comp) is None:
                    if self.do_inplace == True:
                        self.compartments[dest_comp].assign(self.compartments[dest_comp] + deposit)
                    else:
                        self.compartments[dest_comp] = (deposit + self.compartments[dest_comp])

            if injection_table.get("z") is None:
                # core logic for the (node-internal) dendritic calculation
                dz_bu = self.compartments["dz_bu"]
                dz_td = self.compartments["dz_td"]
                z = self.compartments["z"]
                dz = None
                if self.use_dfx is True:
                    dz = dz_td + (dz_bu * self.dfx(z)) # (dz_td + dz_bu) * self.dfx(z)
                else:
                    dz = dz_td + dz_bu

                z_prior = 0.0
                if self.prior_type is not None:
                    if self.lmbda > 0.0:
                        if self.prior_type == "laplace":
                            z_prior = -tf.math.sign(z) * self.lmbda
                        elif self.prior_type == "exp": # Exp: x ~ -exp(-x^2)
                            z_prior = -(tf.math.exp(-tf.math.square(z)) * z * 2) * self.lmbda
                        elif self.prior_type == "cauchy":  # Cauchy: x ~ (1.0 + tf.math.square(z))
                            z_prior = -(z * (2 * self.lmbda))/(1.0 + tf.math.square(z))
                        elif self.prior_type == "gaussian":
                            z_prior = -z * (2 * self.lmbda)
                if self.integrate_type == "euler":
                    '''
                    Euler integration step (under NGC inference dynamics)

                    Constants/meta-parameters:
                    beta - strength of update to node state z
                    leak - controls strength of leak variable/decay
                    zeta - if set to 0 turns off recurrent carry-over of node's current state value
                    prior(z) - distributional prior placed over z (such as kurtotic prior, e.g. Laplace/Cauchy)

                    Dynamics Equation:
                    z <- z * zeta + ( dz * beta - z * leak + prior(z) )
                    '''
                    dz = dz - z * self.leak + z_prior
                    z = z * self.zeta + dz * self.beta
                    if self.threshold_type == "soft_threshold":
                        z = transform_utils.threshold_soft(z, self.thr_lmbda)
                    elif self.threshold_type == "cauchy_threshold":
                        z = transform_utils.threshold_cauchy(z, self.thr_lmbda)

                else:
                    tf.print("Error: Node {0} does not support {1} integration".format(self.name, self.integrate_type))
                    sys.exit(1)
                if injection_table.get("z") is None:
                    if self.do_inplace == True:
                        self.compartments["z"].assign(z)
                    else:
                        self.compartments["z"] = z
            ########################################################################
        # else, skip this core "chunk of computation" if externally set

        if injection_table.get("phi(z)") is None: # apply post-activation non-linearity
            phi_z = None
            if self.n_winners > 0:
                phi_z = self.fx(self.compartments["z"],K=self.n_winners)
            else:
                phi_z = self.fx(self.compartments["z"])
            if self.do_inplace == True:
                self.compartments["phi(z)"].assign(phi_z)
            else:
                self.compartments["phi(z)"] = (phi_z)

        if injection_table.get("S(z)") is None: # apply stochastic/sampling function
            S_z = self.sfx(self.compartments["phi(z)"])
            if self.do_inplace == True:
                self.compartments["S(z)"].assign(S_z)
            else:
                self.compartments["S(z)"] = (S_z)

        if bmask is not None: # applies mask to all component variables of this node
            for key in self.compartments:
                if self.compartments.get(key) is not None:
                    if self.do_inplace == True:
                        self.compartments[key].assign( self.compartments.get(key) * bmask )
                    else:
                        self.compartments[key] = ( self.compartments.get(key) * bmask )

        ########################################################################
        if skip_core_calc == False:
            self.t = self.t + 1

        # a node returns a list of its named component values
        values = []
        injected = []
        for comp_name in self.compartments:
            comp_value = self.compartments.get(comp_name)
            values.append((self.name, comp_name, comp_value))
        return values
