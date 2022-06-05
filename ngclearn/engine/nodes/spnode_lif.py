import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils as transform

class SpNode_LIF(Node):
    """
    | Implements a leaky-integrate and fire (LIF) spiking state node that follows NGC settling dynamics
    | according to:
    |   Jz = dz  OR  d.Jz/d.t = (-Jz + dz) * (dt/tau_curr) IF zeta > 0 // current
    |   d.Vz/d.t = (-Vz + Jz * R) * (dt/tau_mem) // voltage
    |   spike(t) = spike_response_model(Jz(t), Vz(t), ref(t)...) // spikes computed according to SRM
    |   trace(t) = (trace(t-1) * alpha) * (1 - Sz(t)) + Sz(t)  // variable trace filter
    | where:
    |   Jz - current value of the electrical current input to the spiking neurons w/in this node
    |   Vz - current value of the membrane potential of the spiking neurons w/in this node
    |   Sz - the spike signal reading(s) of the spiking neurons w/in this node
    |   dz - aggregated input signals from other nodes/locations to drive current Jz
    |   ref - the current value of the refractory variables (accumulates with time and forces neurons to rest)
    |   alpha - variable trace's interpolation constant (dt/tau <-- input by user)
    |   tau_mem - membrane potential time constant (R_m * C_m  - resistance times capacitance)
    |   tau_curr - electrical current time constant strength
    |   dt - the integration time constant d.t
    |   R - neural membrane resistance

    | Note that the above is used to adjust neural electrical current values via an integator inside a node.
        For example, if the standard/default Euler integrator is used then the neurons inside this
        node are adjusted per step as follows:
    |   Jz <- Jz + d.Jz/d.t // <-- only if zeta > 0
    |   Vz <- Vz + d.Vz/d.t
    |   ref <- ref + d.t (resets to 0 after 1 millisecond)

    | Compartments:
    |   * dz_td - the top-down pressure compartment (deposited signals summed)
    |   * dz_bu - the bottom-up pressure compartment, potentially weighted by phi'(x)) (deposited signals summed)
    |   * Jz - the neural electrical current values
    |   * Vz - the neural membrane potential values
    |   * Sz - the current spike values (binary vector signal) at time t
    |   * Trace_z - filtered trace values of the spike values (real-valued vector)
    |   * ref - the refractory variables (an accumulator)
    |   * mask - a binary mask to be applied to the neural activities
    |   * t_spike - tracks time of last spike (1 slot per neuron in this node)

    | Constants:
    |   * V_thr - voltage threshold (to generate a spike)
    |   * dt - the integration time constant (milliseconds)
    |   * R - the neural membrane resistance (mega Ohms)
    |   * C - the neural membrane capacitance (microfarads)
    |   * tau_m - the membrane potential time constant (tau_m = R * C)
    |   * tau_c - the electrial current time constant (if zeta > 0)
    |   * trace_alpha - the trace variable's interpolation constant
    |   * ref_T - the length of the absolute refractory period (milliseconds)

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        batch_size: batch-size this node should assume (for use with static graph optimization)

        integrate_kernel: Dict defining the neural state integration process type. The expected keys and
            corresponding value types are specified below:

            :`'integrate_type'`: type integration method to apply to neural activity over time.
                If "euler" is specified, Euler integration will be used (future ngc-learn versions will support
                "midpoint"/other methods).

            :`'use_dfx'`: <UNUSED>

            :`'dt'`: type integration time constant for the spiking neurons

            :Note: specifying None will automatically set this node to use Euler integration with dt = 0.25 ms

        spike_kernel: Dict defining the properties of the spiking process. The expected keys and
            corresponding value types are specified below:

            :`'V_thr'`: the (constant) voltage threshold a neuron must cross to spike

             :`'zeta'`: a trigger variable - if > 0, electrical current will be integrated over as well

             :`'tau_mem'`: the membrane potential time constant

             :`'tau_curr'`: the electrical current time constant (only used if zeta > 0, otherwise ignored)

        trace_kernel: Dict defining the signal tracing process type. The expected keys and
            corresponding value types are specified below:

            :`'dt'`: type integration time constant for the trace

            :`'tau'`: the filter time constant for the trace

            :Note: specifying None will automatically set this node to not use variable tracing
    """
    def __init__(self, name, dim, batch_size=1, integrate_kernel=None,
                 spike_kernel=None, trace_kernel=None):
        node_type = "spike_lif_state"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.is_clamped = False

        self.integrate_kernel = integrate_kernel
        self.use_dfx = False
        self.integrate_type = "euler" # Default = euler
        self.dt = 0.25 # integration time constant (ms)
        if integrate_kernel is not None:
            self.use_dfx = integrate_kernel.get("use_dfx")
            self.integrate_type = integrate_kernel.get("integrate_type")
            if integrate_kernel.get("dt") is not None:
                self.dt = integrate_kernel.get("dt") # trace integration time constant (ms)

        self.zeta = 0
        #self.conduct_leak = leak
        # spiking neuron settings
        self.max_spike_rate = 640.0 # 64 Hz is a good default standard value
        self.V_thr = 1.0  # threshold for a neuron's voltage to reach before spiking
        self.R_m = 5.1 # in mega ohms
        self.C_m = 5e-3 # in picofarads

        self.tau_m = self.R_m * self.C_m #20.0 # should be -> tau_mem = R*C
        self.tau_c = 1.0
        self.ref_T = 1.0 # ms

        self.spike_kernel = spike_kernel
        if self.spike_kernel is not None:
            #self.max_spike_rate = self.spike_kernel.get("max_spike_rate")
            if self.spike_kernel.get("V_thr") is not None:
                self.V_thr = self.spike_kernel.get("V_thr")
            if self.spike_kernel.get("R") is not None:
                self.R_m = self.spike_kernel.get("R")
            if self.spike_kernel.get("C") is not None:
                self.C_m = self.spike_kernel.get("C")
            if self.spike_kernel.get("zeta") is not None:
                self.zeta = self.spike_kernel.get("zeta")
            if self.spike_kernel.get("tau_curr") is not None:
                self.tau_c = self.spike_kernel.get("tau_curr")
            if self.spike_kernel.get("ref_T") is not None:
                self.ref_T = self.spike_kernel.get("ref_T")
            if self.spike_kernel.get("tau_mem") is not None:
                self.tau_m = self.spike_kernel.get("tau_mem")
                self.R_m = 1.0
                self.C_m = self.tau_m # C = tau_m/(R = 1)


        self.trace_kernel = trace_kernel
        self.trace_dt = 1.0
        if self.trace_kernel is not None:
            if self.trace_kernel.get("dt") is not None:
                self.trace_dt = self.trace_kernel.get("dt") # trace integration time constant (ms)
            #5.0 # filter time constant -- where dt (or T) = 0.001 (to model ms)
            self.tau = self.trace_kernel.get("tau")

        # derived settings that are a function of other spiking neuron settings
        self.a = np.exp(-self.trace_dt/self.tau)
        #self.tau_m = self.R_m * self.C_m
        #self.kappa = 1.0 #np.exp(-self.dt/self.tau_j)
        #self.kappa = 0.2

        # set LIF spiking neuron-specific (vector/scalar) constants
        self.constant_name = ["V_thr", "dt", "R", "C", "tau_m", "tau_c", "trace_alpha"]
        self.constants = {}
        self.constants["V_thr"] = tf.ones([1,self.dim]) * self.V_thr
        self.constants["dt"] = self.dt
        self.constants["R"] = self.R_m
        self.constants["C"] = self.C_m
        self.constants["tau_m"] = self.tau_m
        self.constants["tau_c"] = self.tau_c
        self.constants["trace_alpha"] = self.a
        self.constants["eps"] = 1e-4
        self.constants["zeta"] = self.zeta
        self.constants["ref_T"] = self.ref_T

        # set LIF spiking neuron-specific vector statistics
        self.compartment_names = ["dz_bu", "dz_td", "Jz", "Vz", "Sz", "Trace_z",
                                  "ref", "t_spike"] #, "x_tar", "Ns"]
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
        info["integration.form"] = self.integrate_kernel
        if self.spike_kernel is not None:
            info["spike.form"] = self.spike_kernel
        if self.trace_kernel is not None:
            info["trace.form"] = self.trace_kernel
        return info

    def step(self, injection_table=None, skip_core_calc=False):
        if injection_table is None:
            injection_table = {}

        bmask = self.masks.get("mask")
        ########################################################################
        if skip_core_calc == False:
            ##########################################################################
            #### update trace variable ####
            trace_alpha = self.constants.get("trace_alpha")
            Sz = self.compartments.get("Sz")
            trace_z_tm1 = self.compartments.get("Trace_z")
            # apply variable trace filters z_l(t) = (alpha * z_l(t))*(1−s(t)) + s_l(t)
            #trace_z = tf.add((trace_z_tm1 * trace_alpha) * (-Sz + 1.0), Sz)
            trace_z = (trace_z_tm1 * trace_alpha) * (-Sz + 1.0) + Sz
            if injection_table.get("Trace_z") is None:
                if self.do_inplace == True:
                    self.compartments["Trace_z"].assign(trace_z)
                else:
                    self.compartments["Trace_z"] = trace_z

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

            # get constants
            # tau_mem = R*C where R = 5.1, C = 5e-3
            eps = self.constants.get("eps") # <-- simulation constant (for refractory variable)
            R = self.constants.get("R")
            C = self.constants.get("C")
            tau_mem = self.constants.get("tau_m") # membrane time constant (R * C)
            V_thr = self.constants.get("V_thr") # get voltage threshold (constant)
            dt = self.constants.get("dt") # integration time constant
            ref_T = self.constants.get("ref_T")
            # get current compartment values
            Jz = self.compartments.get("Jz") # current I
            Vz = self.compartments.get("Vz") # voltage V
            ref = self.compartments.get("ref") # refractory variable

            self.t = self.t + dt # advance time forward by dt (t <- t + dt)

            dz_bu = self.compartments["dz_bu"]
            dz_td = self.compartments["dz_td"]
            dz = dz_td + dz_bu # gather pre-synaptic signals to modify current

            if injection_table.get("Jz") is None:
                zeta = self.constants["zeta"]
                if zeta > 0.0:
                    # integrate over current
                    #Jz = Jz * self.zeta + dz * self.kappa - (Jz * self.conduct_leak) * self.kappa
                    tau_curr = self.constants.get("tau_c") #R * C
                    Jz = Jz + (-Jz + dz) * (dt/tau_curr)
                else:
                    Jz = dz # input current
            ####################################################################
            #### Run LIF spike response model ####
            '''
            Apply the leaky integrate-and-fire spike-response model (SRM LIF):
            V(t + dt) = V(t) + ( -V(t) * leak_lvl + I(t) ) * (dt / tau_m), w/ tau_m = R_m * C_m
            '''
            # update the membrane potential given input current and spike
            if ref_T > 0.0:
                ref = ref + tf.cast(tf.greater(ref, 0.0),dtype=tf.float32) * dt
                # if ref > 1, then ref <- 0
                mask = tf.cast(tf.greater(ref, ref_T),dtype=tf.float32)
                ref = ref * (1.0 - mask)

                # if ref > 0.0: Vz <- 0, else: update
                mask = tf.cast(tf.greater(ref, 0.0),dtype=tf.float32) # ref1 > 0.0
                Vz = (Vz + (-Vz + Jz * R) * (dt/tau_mem)) * (1.0 - mask)

                mask = tf.cast(tf.greater(Vz, V_thr),dtype=tf.float32) # h1 > h_thr
                ref = mask * eps + (1.0 - mask) * ref

                #if ref1 > 0.0: a <- 1, else 0
                Sz = tf.cast(tf.greater(ref, 0.0),dtype=tf.float32)
                #Sz = tf.cast(tf.math.greater_equal(Vz, V_thr), dtype=tf.float32)
            else: # special mode where refractory period is exactly 0, so instantaneous refraction
                Vz = (Vz + (-Vz + Jz * R) * (dt/tau_mem)) #- (Sz * V_thr)
                Sz = tf.cast(tf.greater(Vz, V_thr),dtype=tf.float32)
                Vz = Vz * (-Sz + 1.0)
            ####################################################################

            ####################################################################
            if injection_table.get("Jz") is None:
                if self.do_inplace == True:
                    self.compartments["Jz"].assign(Jz)
                else:
                    self.compartments["Jz"] = Jz
            if injection_table.get("ref") is None:
                if self.do_inplace == True:
                    self.compartments["ref"].assign(ref)
                else:
                    self.compartments["ref"] = ref
            if injection_table.get("Vz") is None:
                if self.do_inplace == True:
                    self.compartments["Vz"].assign(Vz)
                else:
                    self.compartments["Vz"] = Vz
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
            # ##########################################################################
            # #### update trace variable ####
            # trace_alpha = self.constants.get("trace_alpha")
            # trace_z_tm1 = self.compartments.get("Trace_z")
            # # apply variable trace filters z_l(t) = (alpha * z_l(t))*(1−s(t)) + s_l(t)
            # #trace_z = tf.add((trace_z_tm1 * trace_alpha) * (-Sz + 1.0), Sz)
            # trace_z = (trace_z_tm1 * trace_alpha) * (-Sz + 1.0) + Sz
            # if injection_table.get("Trace_z") is None:
            #     if self.do_inplace == True:
            #         self.compartments["Trace_z"].assign(trace_z)
            #     else:
            #         self.compartments["Trace_z"] = trace_z
            # currently un-used (below) target trace variable code
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
        # if skip_core_calc == False:
        #     dt = self.constants.get("dt") # integration time constant
        #     self.t = self.t + dt

        # a node returns a list of its named component values
        values = []
        injected = []
        for comp_name in self.compartments:
            comp_value = self.compartments.get(comp_name)
            values.append((self.name, comp_name, comp_value))
        return values
