import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import stat_utils as stat

class SpNode_Enc(Node):
    """
    | Implements a simple spiking state node that converts its real-valued input
    | vector into an on-the-fly generated Poisson spike train. To control the
    | firing frequency of the spiking neurons within this model, modify the
    | gain parameter (range [0,1]) -- for example, on pixel data normalized
    | to the range of [0,1], setting the gain to 0.25 will result in a firing
    | frequency of approximately 63.75 Hertz (Hz). Note that for real-valued data
    | which should be normalized to the range of [0,1], the actual values of each
    | dimension will be used to dictate specific spiking rates (each dimension
    | spikes in proportion to its feature value/probability).

    | Compartments:
    |   * z - the real-valued input variable to convert to spikes (should be clamped)
    |   * Sz - the current spike values (binary vector signal) at time t
    |   * Trace_z - filtered trace values of the spike values (real-valued vector)
    |   * mask - a binary mask to be applied to the neural activities

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        leak: strength of the conductance leak applied to each neuron's current Jz (Default = 0)

        batch_size: batch-size this node should assume (for use with static graph optimization)

        trace_kernel: Dict defining the signal tracing process type. The expected keys and
            corresponding value types are specified below:

            :`'dt'`: type integration time constant for the trace

            :`'tau'`: the filter time constant for the trace

            :Note: specifying None will automatically set this node to not use variable tracing
    """
    def __init__(self, name, dim, gain=1.0, batch_size=1, trace_kernel=None):
        node_type = "spike_enc_state"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.is_clamped = False

        self.gain = gain
        self.dt = 1.0 # integration time constant (ms)

        self.trace_kernel = trace_kernel
        self.trace_dt = 1.0
        if self.trace_kernel is not None:
            # trace integration time constant (ms)
            self.trace_dt = self.trace_kernel.get("dt")
            # filter time constant
            self.tau = self.trace_kernel.get("tau")

        # derived settings that are a function of other spiking neuron settings
        self.a = np.exp(-self.trace_dt/self.tau)
        self.tau_j = 1.0

        # set LIF spiking neuron-specific (vector/scalar) constants
        self.constant_name = ["gain", "dt", "trace_alpha"]
        self.constants = {}
        self.constants["dt"] = self.dt
        self.constants["gain"] = self.gain
        self.constants["trace_alpha"] = self.a

        # set LIF spiking neuron-specific vector statistics
        self.compartment_names = ["z", "Sz", "Trace_z"] #, "x_tar", "Ns"]
        self.compartments = {}
        for name in self.compartment_names:
            self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]),
                                                  name="{}_{}".format(self.name, name))
        self.mask_names = ["mask"]
        self.masks = {}
        for name in self.mask_names:
            self.masks[name] = tf.Variable(tf.ones([batch_size,dim]),
                                           name="{}_{}".format(self.name, name))

        self.connected_cables = []

    def compile(self):
        info = super().compile()
        #info["leak"] = self.leak
        if self.trace_kernel is not None:
            info["trace.form"] = self.trace_kernel
        return info

    def step(self, injection_table=None, skip_core_calc=False):
        if injection_table is None:
            injection_table = {}

        bmask = self.masks.get("mask")
        ########################################################################
        if skip_core_calc == False:
            z = self.compartments.get("z")
            #Sz = transform.convert_to_spikes(z, self.max_spike_rate, self.dt)
            Sz = stat.convert_to_spikes(z, gain=self.gain)

            if injection_table.get("Sz") is None:
                if self.do_inplace == True:
                    self.compartments["Sz"].assign(Sz)
                else:
                    self.compartments["Sz"] = Sz
            ##########################################################################

            ##########################################################################
            trace_alpha = self.constants.get("trace_alpha")
            trace_z_tm1 = self.compartments.get("Trace_z")
            # apply variable trace filters z_l(t) = (alpha * z_l(t))*(1−s`(t)) +s_l(t)
            trace_z = tf.add((trace_z_tm1 * trace_alpha) * (-Sz + 1.0), Sz)
            if injection_table.get("Trace_z") is None:
                if self.do_inplace == True:
                    self.compartments["Trace_z"].assign(trace_z)
                else:
                    self.compartments["Trace_z"] = trace_z
            # Ns = self.compartments.get("Ns")
            # x_tar = self.compartments.get("x_tar")
            # x_tar = x_tar + (trace_z - x_tar)/Ns
            # if injection_table.get("x_tar") is None:
            #     if self.do_inplace == True:
            #         self.compartments["x_tar"].assign(x_tar)
            #     else:
            #         self.compartments["x_tar"] = x_tar
            # Ns = Ns + 1
            # if injection_table.get("Ns") is None:
            #     if self.do_inplace == True:
            #         self.compartments["Ns"].assign(Ns)
            #     else:
            #         self.compartments["Ns"] = Ns

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
