import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.spiking.spnode_lif import SpNode_LIF
from ngclearn.utils import transform_utils as transform

class SpNode_ELIF(SpNode_LIF): # inherits from SpNode_LIF class
    """
    | Implements the exponential leaky-integrate and fire (ELIF) spiking state
    | node that follows NGC settling dynamics according to:
    |   Jz = dz  OR  d.Jz/d.t = (-Jz + dz) * (dt/tau_curr) IF zeta > 0 // current
    |   d.Vz/d.t = (V_rest - Vz + Jz * R + exp((Vz - V_rh)/Delta_T ) * Delta_T) * (dt/tau_mem) // voltage
    |   Sz = spike(t) = spike_response_model(Jz(t), Vz(t), ref(t)...) // spikes computed according to SRM
    |   d.Trace_z/d.t = -Trace_z/tau_trace + (Sz * (ref <= eps))  // variable trace filter
    |   d.V_delta/d.t = -(V - 0.0)/tau_A + (Sz * (ref <= eps)) * A_theta // voltage delta
    | where:
    |   Jz - current value of the electrical current input to the spiking neurons w/in this node
    |   Vz - current value of the membrane potential of the spiking neurons w/in this node
    |   Sz - the spike signal reading(s) of the spiking neurons w/in this node
    |   dz - aggregated input signals from other nodes/locations to drive current Jz
    |   ref - the current value of the refractory variables (accumulates with time and forces neurons to rest)
    |   V_delta - the current value of the voltage threshold adaptation delta
    |   alpha - variable trace's interpolation constant (dt/tau <-- input by user)
    |   tau_mem - membrane potential time constant (R_m * C_m  - resistance times capacitance)
    |   tau_curr - electrical current time constant strength
    |   dt - the integration time constant d.t
    |   R - neural membrane resistance
    |   V_rest - resting membrane potential
    |   V_rh - rheobase threshold
    |   Delta_T - sharpness control meta-parameter

    | Note that the above is used to adjust neural electrical current values via an integator inside a node.
        For example, the standard/default Euler integrator is used and the neurons inside this
        node are adjusted per step as follows:
    |   Jz <- Jz + d.Jz/d.t // <-- only if zeta > 0
    |   Vz <- Vz + d.Vz/d.t
    |   ref <- ref + d.t (resets to 0 after 1 millisecond)
    |   Trace_z <- Trace_z + d.Trace_z/d.t
    |   V_delta <- V_delta + d.V_delta/d.t. // leads to V_thr(t) = V_thr + V_delta(t)

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
    |   * V_delta - the adaptation of the voltage threshold over time

    | Constants:
    |   * V_thr - voltage threshold (to generate a spike)
    |   * Delta_T - sharpness parameter
    |   * V_rh - rheobase threshold or ("internal voltage" threshold for exponential term)
    |   * V_reset - voltage/membrane potential reset value
    |   * V_rest - resting membrane potential
    |   * dt - the integration time constant (milliseconds)
    |   * R - the neural membrane resistance (mega Ohms)
    |   * C - the neural membrane capacitance (microfarads)
    |   * tau_m - the membrane potential time constant (tau_m = R * C)
    |   * tau_c - the electrial current time constant (if zeta > 0)
    |   * trace_tau - the trace variable's interpolation time constant
    |   * ref_T - the length of the absolute refractory period (milliseconds)
    |   * tau_A - the voltage adaptation time constant
    |   * A_theta - the voltage adaptation increment

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

             :Note: more constants can be set through this kernel (see above "Constants" for values to set)

        trace_kernel: Dict defining the signal tracing process type. The expected keys and
            corresponding value types are specified below:

            :`'dt'`: type integration time constant for the trace

            :`'tau_trace'`: the filter time constant for the trace

            :Note: specifying None will automatically set this node to not use variable tracing
    """
    def __init__(self, name, dim, batch_size=1, integrate_kernel=None,
                 spike_kernel=None, trace_kernel=None):
        node_type = "spike_elif_state"
        super().__init__(name, dim, batch_size, integrate_kernel,
                         spike_kernel, trace_kernel)

        # additional constants to set up for Exp-LIF model
        if self.constants.get("V_rh") is None:
            self.constants["V_rh"] = self.constants["V_thr"] - 0.2
        if self.constants.get("Delta_T") is None:
            self.constants["Delta_T"] = 1.0
        if self.spike_kernel is not None:
            if self.spike_kernel.get("V_reset") is None:
                self.constants["V_reset"] = -0.1 # voltage reset value

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
            V_reset = self.constants.get("V_reset") # reset voltage
            V_rest = self.constants.get("V_rest") # resting voltage
            dt = self.constants.get("dt") # integration time constant
            ref_T = self.constants.get("ref_T") # refrectory time
            tau_A = self.constants.get("tau_A") # voltage delta time constant
            A_theta = self.constants.get("A_theta") # voltage delta increment
            Delta_T = self.constants.get("Delta_T")
            V_rh = self.constants.get("V_rh") # get rheobase threshold

            # get relevant compartment values
            t_spike = self.compartments.get("t_spike")
            V_delta = self.compartments.get("V_delta") # get voltage threshold delta
            Jz = self.compartments.get("Jz") # current I
            Vz = self.compartments.get("Vz") # voltage V
            ref = self.compartments.get("ref") # refractory variable

            V_thr = V_thr + V_delta # update adaptive voltage threshold

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
            #### Run ELIF spike response model ####
            '''
            Apply exponential leaky integrate-and-fire spike-response model (SRM ELIF):
            '''
            smask = 1.0
            mask = 0.0
            if ref_T > 0.0: # calculate refractory state variables
                ref = ref + tf.cast(tf.greater(ref, 0.0),dtype=tf.float32) * dt
                mask = tf.cast(tf.greater(ref, ref_T),dtype=tf.float32)
                ref = ref * (1.0 - mask)
                mask = tf.cast(tf.greater(ref, 0.0),dtype=tf.float32)

            exp_term = tf.math.exp((Vz - V_rh)/Delta_T) * Delta_T # calc exponential term
            Vz = (Vz + (V_rest - Vz + exp_term + Jz * R) * (dt/tau_mem)) #- (Sz * V_thr)
            Vz = Vz * (1.0 - mask) + (mask * V_reset) # refraction keeps voltage at rest if > T_ref

            Sz = tf.cast(tf.greater(Vz, V_thr),dtype=tf.float32)
            if ref_T > 0.0:
                ref = ref * mask + (Sz * eps) * (1.0 - mask)

            t_spike = t_spike * (1.0 - Sz) + (Sz * self.t) # record spike time
            V = Vz + 0 # store membrane potential before it is potentially reset
            Vz = Vz * (1.0 - Sz) + Vz * (Sz * V_reset)

            d_Vdelta_d_t = (0.0 - V_delta)/tau_A + Sz * A_theta
            V_delta = V_delta + d_Vdelta_d_t # update voltage threshold

            #### update trace variable ####
            tau_tr = self.constants.get("tau_trace")
            tr_z = self.compartments.get("Trace_z")
            d_tr = -tr_z/tau_tr + Sz
            tr_z = tr_z + d_tr

            if ref_T > 0.0:
                ## persistent spike signals - helps with learning stability
                # we register a spike as 1 also if refractory period has begun
                # since a spike signal/trace could persist after its initial emission
                Sz = tf.cast(tf.greater(ref, 0.0),dtype=tf.float32)

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
            if injection_table.get("V_delta") is None:
                if self.do_inplace == True:
                    self.compartments["V_delta"].assign(V_delta)
                else:
                    self.compartments["V_delta"] = V_delta
            if injection_table.get("V") is None:
                if self.do_inplace == True:
                    self.compartments["V"].assign(V)
                else:
                    self.compartments["V"] = V
            if injection_table.get("Vz") is None:
                if self.do_inplace == True:
                    self.compartments["Vz"].assign(Vz)
                else:
                    self.compartments["Vz"] = Vz
            if injection_table.get("Sz") is None:
                if self.do_inplace == True:
                    self.compartments["Sz"].assign(Sz)
                    self.compartments["t_spike"].assign(t_spike)
                else:
                    self.compartments["Sz"] = Sz
                    self.compartments["t_spike"] = t_spike
            if injection_table.get("Trace_z") is None:
                if self.do_inplace == True:
                    self.compartments["Trace_z"].assign(tr_z)
                else:
                    self.compartments["Trace_z"] = tr_z
            # ##########################################################################

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
