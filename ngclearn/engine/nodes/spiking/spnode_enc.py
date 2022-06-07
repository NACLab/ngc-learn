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
    |   * t_spike - time of last spike (per neuron inside this node)

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        leak: strength of the conductance leak applied to each neuron's current Jz (Default = 0)

        batch_size: batch-size this node should assume (for use with static graph optimization)

        integrate_kernel: Dict defining the neural state integration process type. The expected keys and
            corresponding value types are specified below:

            :`'integrate_type'`: <UNUSED>

            :`'dt'`: type integration time constant for the spiking neurons

        spike_kernel: <UNUSED>

        trace_kernel: Dict defining the signal tracing process type. The expected keys and
            corresponding value types are specified below:

            :`'dt'`: type integration time constant for the trace

            :`'tau_trace'`: the filter time constant for the trace

            :Note: specifying None will automatically set this node to not use variable tracing
    """
    def __init__(self, name, dim, gain=1.0, batch_size=1, integrate_cfg=None,
                 spike_kernel=None, trace_kernel=None):
        node_type = "spike_enc_state"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.is_clamped = False

        self.gain = gain
        self.dt = 1.0 # integration time constant (ms)
        self.tau_m = 1.0
        self.integrate_cfg = integrate_cfg
        if self.integrate_cfg is not None:
            self.dt = self.integrate_cfg.get("dt")

        self.trace_kernel = trace_kernel
        self.trace_dt = 1.0
        self.tau_trace = 20.0
        if self.trace_kernel is not None:
            if self.trace_kernel.get("dt") is not None:
                self.trace_dt = self.trace_kernel.get("dt") # trace integration time constant (ms)
            #5.0 # filter time constant -- where dt (or T) = 0.001 (to model ms)
            self.tau_trace = self.trace_kernel.get("tau_trace")

        # set LIF spiking neuron-specific (vector/scalar) constants
        self.constant_name = ["gain", "dt", "tau_trace"]
        self.constants = {}
        self.constants["dt"] = self.dt
        self.constants["gain"] = self.gain
        self.constants["tau_trace"] = self.tau_trace

        # set LIF spiking neuron-specific vector statistics
        self.compartment_names = ["z", "Sz", "Trace_z", "t_spike", "ref"]
        self.compartments = {}
        for name in self.compartment_names:
            self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]),
                                                  name="{}_{}".format(self.name, name))
        self.mask_names = ["mask"]
        self.masks = {}
        for name in self.mask_names:
            self.masks[name] = tf.Variable(tf.ones([batch_size,dim]),
                                           name="{}_{}".format(self.name, name))

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
            # compute spike response model
            dt = self.constants.get("dt") # integration time constant
            z = self.compartments.get("z")
            self.t = self.t + dt # advance time forward by dt (t <- t + dt)
            #Sz = transform.convert_to_spikes(z, self.max_spike_rate, self.dt)
            Sz = stat.convert_to_spikes(z, gain=self.gain)

            if injection_table.get("Sz") is None:
                if self.do_inplace == True:
                    t_spike = self.compartments.get("t_spike")
                    t_spike = t_spike * (1.0 - Sz) + (Sz * self.t)
                    self.compartments["Sz"].assign(Sz)
                    self.compartments["t_spike"].assign(t_spike)
                else:
                    t_spike = self.compartments.get("t_spike")
                    t_spike = t_spike * (1.0 - Sz) + (Sz * self.t)
                    self.compartments["Sz"] = Sz
                    self.compartments["t_spike"] = t_spike

            #### update trace variable ####
            tau_tr = self.constants.get("tau_trace")
            #Sz = self.compartments.get("Sz")
            tr_z = self.compartments.get("Trace_z")
            d_tr = -tr_z/tau_tr + Sz
            tr_z = tr_z + d_tr
            if injection_table.get("Trace_z") is None:
                if self.do_inplace == True:
                    self.compartments["Trace_z"].assign(tr_z)
                else:
                    self.compartments["Trace_z"] = tr_z
            ##########################################################################

            ##########################################################################
            # trace_alpha = self.constants.get("trace_alpha")
            # trace_z_tm1 = self.compartments.get("Trace_z")
            # # apply variable trace filters z_l(t) = (alpha * z_l(t))*(1−s`(t)) +s_l(t)
            # trace_z = tf.add((trace_z_tm1 * trace_alpha) * (-Sz + 1.0), Sz)
            # if injection_table.get("Trace_z") is None:
            #     if self.do_inplace == True:
            #         self.compartments["Trace_z"].assign(trace_z)
            #     else:
            #         self.compartments["Trace_z"] = trace_z

        if bmask is not None: # applies mask to all component variables of this node
            for key in self.compartments:
                if self.compartments.get(key) is not None:
                    if self.do_inplace == True:
                        self.compartments[key].assign( self.compartments.get(key) * bmask )
                    else:
                        self.compartments[key] = ( self.compartments.get(key) * bmask )

        ########################################################################

        # a node returns a list of its named component values
        values = []
        injected = []
        for comp_name in self.compartments:
            comp_value = self.compartments.get(comp_name)
            values.append((self.name, comp_name, comp_value))
        return values
