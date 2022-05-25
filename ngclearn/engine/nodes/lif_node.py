import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils

class LIFNode(Node):
    """
    | Implements a leaky-integrate and fire (LIF) spiking state node that follows NGC settling dynamics
    | according to:
    |   d.Jz/d.t = dz * kappa - ( Jz * conduct_leak ) * kappa // current
    |   d.Vz/d.t = (-Vz * membrane_leak + Jz) * (dt/tau_m) // voltage
    |   spike(t) = spike_response_model(Jz(t), Vz(t),...) // spikes computed according to SRM
    |   trace(t) = (trace(t-1) * alpha) * (1 - spike(t)) + spike(t)  // variable filter
    | where:
    |   dz - aggregated input signals from other nodes/locations
    |   tau_m - R_m * C_m  (resistance times capacitance)
    |   kappa - strength of update to node electrical current Jz (dt/tau_j)
    |   conduct_leak - controls strength of current leak/decay
    |   membrane_leak - controls strength of voltage leak/decay
    |   alpha - variable trace's interpolation constant (dt/tau)

    | Note that the above is used to adjust neural electrical current values via an integator inside a node.
        For example, if the standard/default Euler integrator is used then the neurons inside this
        node are adjusted per step as follows:
    |   Jz <- Jz + d.Jz/d.t

    | Compartments:
    |   * dz_td - the top-down pressure compartment (deposited signals summed)
    |   * dz_bu - the bottom-up pressure compartment, potentially weighted by phi'(x)) (deposited signals summed)
    |   * Jz - the neural electrical current values
    |   * Vz - the neural voltage values
    |   * Sz - the current spike values (binary vector signal) at time t
    |   * Trace_z - filtered trace values of the spike values (real-valued vector)
    |   * mask - a binary mask to be applied to the neural activities

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        leak: strength of the conductance leak applied to each neuron's current Jz (Default = 0)

        batch_size: batch-size this node should assume (for use with static graph optimization)

        integrate_kernel: Dict defining the neural state integration process type. The expected keys and
            corresponding value types are specified below:

            :`'integrate_type'`: type integration method to apply to neural activity over time.
                If "euler" is specified, Euler integration will be used (future ngc-learn versions will support
                "midpoint"/other methods).

            :`'use_dfx'`: a boolean that decides if phi'(v) (activation derivative) is used in the integration
                process/update.

            :Note: specifying None will automatically set this node to use Euler integration w/ use_dfx=False

        spike_kernel: TO-WRITE

        trace_kernel: Dict defining the signal tracing process type. The expected keys and
            corresponding value types are specified below:

            :`'dt'`: type integration time constant

            :`'tau'`: the filter time constant

            :`'tau_j'`: affects the eletrical update factor

            :Note: specifying None will automatically set this node to not use variable tracing
    """
    def __init__(self, name, dim, beta=1.0, leak=0.0, batch_size=1,
                 integrate_kernel=None, spike_kernel=None, trace_kernel=None):
        node_type = "lif_state"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.is_clamped = False

        self.integrate_kernel = integrate_kernel
        self.prior_kernel = prior_kernel
        self.threshold_kernel = threshold_kernel
        self.use_dfx = False
        self.integrate_type = "euler" # Default = euler
        self.dt = 1.0 # integration time constant (ms)
        if integrate_kernel is not None:
            self.use_dfx = integrate_kernel.get("use_dfx")
            self.integrate_type = integrate_kernel.get("integrate_type")
            self.dt = integrate_kernel.get("dt") # integration time constant


        # spiking neuron settings
        self.max_spike_rate = 640.0 # 64 Hz is a good default standard value
        self.V_thr = 0.5 #0.4 #1.0  # threshold for a neuron's voltage to reach before spiking
        self.membrane_leak = 1.0 #0.0
        self.conduct_leak = leak #0.0 #1.0
        self.abs_refractory_time = 2.0
        self.R_m = 1.0 # 1 kOhm
        self.C_m = 10.0 # 10 pF
        self.tau_j = 1.0 # ms # electrical current time constant
        self.tau = 5.0 # filter time constant -- where dt (or T) = 0.001 (to model ms)

        self.spike_kernel = spike_kernel
        if self.trace_kernel is not None:
            self.trace_dt = self.spike_kernel.get("max_spike_rate")
            self.V_thr = self.spike_kernel.get("V_thr")
            self.membrane_leak = self.spike_kernel.get("membrane_leak")
            self.abs_refractory_time = self.spike_kernel.get("abs_refractory_time")
            self.tau_j = self.tau_j.get("tau_j")

        self.trace_kernel = trace_kernel
        self.trace_dt = 1.0
        if self.trace_kernel is not None:
            self.trace_dt = self.trace_kernel.get("dt") #1.0 # trace integration time constant (ms)
            self.tau = self.trace_kernel.get("tau") #5.0 # filter time constant -- where dt (or T) = 0.001 (to model ms)
            #self.tau_j = self.trace_kernel.get("tau_j")

        # derived settings that are a function of other spiking neuron settings
        self.a = np.exp(-self.trace_dt/self.tau)
        self.tau_m = self.R_m * self.C_m
        self.kappa = np.exp(-self.dt/self.tau_j)
        #self.kappa = 0.2

        # set LIF spiking neuron-specific (vector/scalar) constants
        self.compartment_names = ["V_thr", "dt", "tau_m", "membrane_leak", "abs_refractory_time"]
        self.constants["V_thr"] = tf.ones([1,self.dim]) * self.V_thr
        self.constants["dt"] = self.dt
        self.constants["tau_m"] = self.tau_m
        self.constants["membrane_leak"] = self.membrane_leak
        self.constants["abs_refractory_time"] = self.abs_refractory_time
        self.constants["trace_alpha"] = self.a

        # set LIF spiking neuron-specific vector statistics
        self.compartment_names = ["dz_bu", "dz_td", "Jz", "Vz", "rfr_z", "Sz", "Trace_z"]
        self.compartments = {}
        for name in self.compartment_names:
            self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name="{}_{}".format(self.name, name))
        self.mask_names = ["mask"]
        self.masks = {}
        for name in self.mask_names:
            self.masks[name] = tf.Variable(tf.ones([batch_size,dim]), name="{}_{}".format(self.name, name))

        self.connected_cables = []

    def compile(self):
        """
        Executes the "compile()" routine for this cable.

        Returns:
            a dictionary containing post-compilation check information about this cable
        """
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

            if injection_table.get("Jz") is None:
                # core logic for the (node-internal) dendritic calculation
                dz_bu = self.compartments["dz_bu"]
                dz_td = self.compartments["dz_td"]
                Jz = self.compartments["Jz"]
                dz = dz_td + dz_bu

                if self.integrate_type == "euler":
                    '''
                    Euler integration step (under NGC inference dynamics for the
                    electrial current that will drive a LIF spike-response model)

                    Constants/meta-parameters:
                    kappa - strength of update to electrial current Jz (function
                            of integration time constant dt)
                    conduct_leak - controls strength of conductance leak/decay

                    Dynamics Equation:
                    Jz <- Jz + dz * kappa - ( Jz * conduct_leak ) * kappa
                    '''
                    # integrate the electrical current J (also applying a conductance leak)
                    Jz = Jz + dz * self.kappa - (Jz * self.conduct_leak) * self.kappa
                    #Jz = Jz - (dz - (Jz * self.conduct_leak)) * self.kappa
                else:
                    tf.print("Error: Node {0} does not support {1} integration".format(self.name, self.integrate_type))
                    sys.exit(1)
                if injection_table.get("Jz") is None:
                    if self.do_inplace == True:
                        self.compartments["Jz"].assign(Jz)
                    else:
                        self.compartments["Jz"] = Jz
            ########################################################################
        # else, skip this core "chunk of computation" if externally set
        '''
        Apply the leaky integrate-and-fire spike-response model (SRM LIF):
        V(t + dt) = V(t) + ( -V(t) * leak_lvl + I(t) ) * (dt / tau_m), where tau_m = R_m * C_m

        Returns:
            (spike_t, volt_t, refractory variable)
        '''
        # get constants
        dt = self.constants.get("dt")
        tau_m = self.constants.get("tau_m")
        membrane_leak = self.constants.get("membrane_leak")
        abs_refractory_time = self.constants.get("abs_refractory_time")
        V_thr = self.constants.get("V_thr") # get voltage threshold (constant)

        # get current relevant compartment states
        Vz = self.compartments.get("Vz")
        rfr_z = self.compartments.get("rfr_z")
        Sz = self.compartments.get("Sz")

        ########################################################################
        ## Calculate the SRM (spike response model)
        rfr_t = tf.nn.relu(tf.subtract(rfr_t, dt))
        # update the voltage variable first
        # V(t + dt) = V(t) + ( -V(t) * leak_lvl + I(t) ) * (dt / tau_m), where tau_m = R_m * C_m
        V_t = tf.add(V_t, tf.add(-V_t * membrane_leak, J_t) * (dt / tau_m))
        # if an absolute refactory period is used, the neuronal SRM must account for this
        V_t = V_t * tf.cast(tf.math.equal(rfr_t, 0.0), dtype=tf.float32)
        # detect a spike and reset to resting potential of 0 if one is to occur
        Sz = spike_t = tf.cast(tf.math.greater_equal(V_t, V_thr), dtype=tf.float32)
        Vz = V_t = tf.subtract(V_t, V_t * spike_t)
        # we update refactory tracking variable if spikes occurred
        rfr_z = rfr_t = tf.add(rfr_t , spike_t * abs_refractory_time)
        ########################################################################
        if injection_table.get("Vz") is None:
            if self.do_inplace == True:
                self.compartments["Vz"].assign(Vz)
            else:
                self.compartments["Vz"] = Vz
        if injection_table.get("rfr_z") is None:
            if self.do_inplace == True:
                self.compartments["rfr_z"].assign(rfr_z)
            else:
                self.compartments["rfr_z"] = rfr_z
        if injection_table.get("Sz") is None:
            if self.do_inplace == True:
                self.compartments["Sz"].assign(Sz)
            else:
                self.compartments["Sz"] = Sz
        ##########################################################################

        ##########################################################################
        if skip_core_calc == False:
            trace_alpha = self.constants.get("trace_alpha")
            trace_z_tm1 = self.compartments.get("Trace_z")
            # apply variable trace filters z_l(t) = (alpha * z_l(t))*(1−s`(t)) +s_l(t)
            trace_z = tf.add((trace_z_tm1 * trace_alpha) * (-spike_t + 1.0), spike_t)
            if injection_table.get("Trace_z") is None:
                if self.do_inplace == True:
                    self.compartments["Trace_z"].assign(trace_z)
                else:
                    self.compartments["Trace_z"] = trace_z

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
