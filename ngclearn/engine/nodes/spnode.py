"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""

import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils

class SpNode(Node):
    """
        Spiking state node (leaky-integrate and fire, LIF)

        @author Alexander G. Ororbia
    """
    def __init__(self, name, dim, integrate_kernel=None, prior_kernel=None,
                 lateral_kernel=None, trace_kernel=None):
        node_type = "spike_state"
        super().__init__(node_type, name, dim)
        self.use_dfx = False
        self.integrate_type = "euler" # euler, midpoint
        if integrate_kernel is not None:
            self.use_dfx = integrate_kernel.get("use_dfx")
            self.integrate_type = integrate_kernel.get("integrate_type")
        self.prior_type = None
        self.lbmda = 0.0
        if prior_kernel is not None:
            self.prior_type = prior_kernel.get("prior_type")
            self.lbmda = prior_kernel.get("lambda")

        # fx, dfx = transform_utils.decide_fun(act_fx)
        # self.fx = fx
        # self.dfx = dfx

        # spiking neuron settings
        self.max_spike_rate = 640.0 #64.0 # 64 Hz is a good default standard value
        self.V_thr = 0.5 #0.4 #1.0  # threshold for a neuron's voltage to reach before spiking --> affects build_RIF
        self.membrane_leak = 1.0 #1.0 #0.0
        self.conduct_leak = 0.0 #0.0 #1.0 # 0.0
        self.abs_refractory_time = 2.0 #2.0 #2.0 #1.0 #0.0 #2.0 #0.0 # ms
        self.R_m = 1.0 # 1 kOhm
        self.C_m = 10.0 # 10 pF
        self.tau_j = 1.0 # ms
        self.tau = 5.0 # filter time constant -- where dt (or T) = 0.001 (to model ms)
        self.dt = 1.0 # integration time constant (ms)

        if trace_kernel is not None:
            self.dt = trace_kernel.get("dt") #1.0 # integration time constant (ms)
            self.tau = trace_kernel.get("tau") #5.0 # filter time constant -- where dt (or T) = 0.001 (to model ms)
            self.tau_j = trace_kernel.get("tau_j")

        # derived settings that are a function of other spiking neuron settings
        self.a = np.exp(-self.dt/self.tau)
        self.tau_m = self.R_m * self.C_m
        self.kappa = np.exp(-self.dt/self.tau_j)
        #self.kappa = 0.2

        # spiking neuron-specific vector statistics
        self.stat["V_thr"] = tf.ones([1,self.dim]) * self.V_thr
        self.stat["phi(z)"] = None
        self.stat["Jz"] = None
        self.stat["Vz"] = None
        self.stat["rfr_z"] = None
        self.stat["Sz"] = None

        self.build_tick()

    def check_correctness(self):
        is_correct = True
        for j in range(len(self.input_nodes)):
            n_j = self.input_nodes[j]
            cable_j = self.input_cables[j]
            dest_var_j = cable_j.out_var
            if dest_var_j == "dz" or dest_var_j == "V_thr" or dest_var_j == "Jz":
                is_correct = True
            else:
                is_correct = False
                print("ERROR: Cable {0} mis-wires to {1}.{2} (can only be .dz)".format(cable_j.name, self.name, dest_var_j))
                break
        return is_correct

    ############################################################################
    # Signal Transmission Routines
    ############################################################################

    def step(self, skip_core_calc=False):
        # get current values of all components of this cell/node
        Jz = self.stat.get("Jz")
        phi_z = self.stat.get("phi(z)")
        Vz = self.stat.get("Vz")
        rfr_z = self.stat.get("rfr_z")
        Sz = self.stat.get("Sz")
        V_thr = self.stat.get("V_thr")

        if self.is_clamped is False and skip_core_calc is False:

            for j in range(len(self.input_nodes)):
                n_j = self.input_nodes[j]
                cable_j = self.input_cables[j]
                dest_var_j = cable_j.out_var
                #print("Parent ",n_j.name)
                # print("     z = ",n_j.extract("z"))
                # print("phi(z) = ",n_j.extract("phi(z)"))
                tick_j = self.tick.get(dest_var_j)
                var_j = self.stat.get(dest_var_j) # get current value of component
                dz_j = cable_j.propagate(n_j)
                # if n_j.name == "m3":
                #     print("Parent ",n_j.name)
                #     print(dz_j[0,:])
                #     sys.exit(0)
                if tick_j > 0: #if var_j is not None:
                    var_j = var_j + dz_j
                else:
                    var_j = dz_j
                self.stat[dest_var_j] = var_j
                self.tick[dest_var_j] = self.tick[dest_var_j] + 1

            V_thr = self.stat.get("V_thr")
            if V_thr is None:
                V_thr = self.V_thr

            dz = self.stat.get("dz")
            if dz is None:
                dz = 0.0
            if self.integrate_type == "euler":
                # integrate the electrical current J (also applying a conductance leak)
                Jz = Jz + dz * self.kappa - (Jz * self.conduct_leak) * self.kappa
                #Jz = Jz - (dz - (Jz * self.conduct_leak)) * self.kappa
            else:
                print("Error: Node {0} does not support {1} integration".format(self.name, self.integrate_type))
                sys.exit(1)

        # the post-activation function is computed always, even if pre-activation is clamped
        #phi_z = self.fx(z)
        self.stat["Jz"] = Jz
        #print("Jz:\n",Jz)
        ##########################################################################
        # compute SRM (spike response model)
        #V_thr = self.V_thr
        Sz, Vz, rfr_z = self.apply_SRM_LIF(Jz, Vz, rfr_z, Sz, self.dt, self.tau_m,
                                              self.membrane_leak, self.abs_refractory_time, V_thr)
        self.stat["Vz"] = Vz
        self.stat["rfr_z"] = rfr_z
        self.stat["Sz"] = Sz
        ##########################################################################

        ##########################################################################
        if self.is_clamped is False and skip_core_calc is False:
            # apply variable trace filters z_l(t) = (alpha * z_l(t))*(1−s`(t)) +s_l(t)
            phi_z = tf.add((phi_z * self.a) * (-Sz + 1.0), Sz)
            self.stat["phi(z)"] = phi_z
        #else:
        #    self.stat["phi(z)"] = phi_z
        ##########################################################################

        self.build_tick()

    #@tf.function
    def apply_SRM_LIF(self, J_t, V_t, rfr_t, spike_t, dt, tau_m, membrane_leak, abs_refractory_time = 0.0, V_thr=0.5):
        """
        Apply the leaky integrate-and-fire spike-response model (SRM LIF):
        V(t + dt) = V(t) + ( -V(t) * leak_lvl + I(t) ) * (dt / tau_m), where tau_m = R_m * C_m

        Returns: (spike_t, volt_t, refractory variable)
        """
        rfr_t = tf.nn.relu(tf.subtract(rfr_t, dt))
        #rfr_t.assign( tf.nn.relu(tf.subtract(rfr_t, dt)) )

        #gate_t = tf.cast(tf.math.equal(rfr_t, 0.0), dtype=tf.float32) # for all decrements == 0, we allow potential build-up

        # update the voltage variable first
        # V(t + dt) = V(t) + ( -V(t) * leak_lvl + I(t) ) * (dt / tau_m), where tau_m = R_m * C_m
        #V_t.assign( tf.add(V_t, tf.add(-V_t * membrane_leak, J_t) * (dt / tau_m)) )
        V_t = tf.add(V_t, tf.add(-V_t * membrane_leak, J_t) * (dt / tau_m))
        # if an absolute refactory period is used, the neuronal SRM must account for this
        V_t = V_t * tf.cast(tf.math.equal(rfr_t, 0.0), dtype=tf.float32)
        #V_t.assign( V_t * tf.cast(tf.math.equal(rfr_t, 0.0), dtype=tf.float32) ) # must gate neurons via refractory variable

        # detect a spike and reset to resting potential of 0 if one is to occur
        spike_t = tf.cast(tf.math.greater_equal(V_t, V_thr), dtype=tf.float32)
        #spike_t.assign( tf.cast(tf.math.greater_equal(V_t, V_thr), dtype=tf.float32) )
        V_t = tf.subtract(V_t, V_t * spike_t)
        #V_t.assign( tf.subtract(V_t, V_t * spike_t) )

        # we update refactory tracking variable if spikes occurred
        rfr_t = tf.add(rfr_t , spike_t * abs_refractory_time)
        #rfr_t.assign( tf.add(rfr_t , spike_t * abs_refractory_time) )

        return spike_t, V_t, rfr_t

    def clear(self):
        self.build_tick()
        self.is_clamped = False
        self.stat["V_thr"] = tf.ones([1,self.dim]) * self.V_thr
        self.stat["Jz"] = None
        self.stat["Vz"] = None
        self.stat["rfr_z"] = None
        self.stat["Sz"] = None
        self.stat["dz"] = None
        self.stat["z"] = None
        self.stat["phi(z)"] = None
