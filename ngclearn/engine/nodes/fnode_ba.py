import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils

class FNode_BA(Node):
    """
    | This is a "teaching" forward node - this node was particularly designed to
    | generate a learning signal for models that are to learn via a broadcast
    | (feedback) alignment approach. (This is a convenience class to support BA.)

    | Implements a feedforward teaching (stateless) transmission node:
    |   z = (dz) * [(Jz > 0) * sech^2(Jz * c2)]  and z = dz if c2 <= 0
    | where:
    |   dz - aggregated input signals from other nodes/locations (typically error nodes)
    |   Jz - input current signal
    |   c2 - a constant value tuned to the sech^2(x) function (this is for a spiking neurons)
    |        note that setting c2 to <= 0, then a non-spiking teaching forward node signal
    |        is created (note if c2 = 0, then this node does not use its Jz compartment)

    | Compartments:
    |   * dz - incoming pressures/signals (deposited signals summed)
    |   * Jz - input current signal to use to trigger spiking BA-learning
    |   * z - the state values/neural activities, set as: z = dz
    |   * phi(z) -  the post-activation of the neural activities

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        act_fx: activation function -- phi(v) -- to apply to neural activities

        batch_size: batch-size this node should assume (for use with static graph optimization)
    """
    def __init__(self, name, dim, act_fx="identity", batch_size=1):
        node_type = "forward_bfa"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.is_clamped = False

        self.act_fx = act_fx
        fx, dfx = transform_utils.decide_fun(act_fx)
        self.fx = fx
        self.dfx = dfx
        self.n_winners = -1
        if "bkwta" in act_fx:
            self.n_winners = int(act_fx[act_fx.index("(")+1:act_fx.rindex(")")])

        self.constant_name = ["c1", "c2"]
        self.constants = {}
        self.constants["c1"] = 0.82
        self.constants["c2"] = 0.08

        self.compartment_names = ["dz", "Jz", "z", "phi(z)"]
        self.compartments = {}
        for name in self.compartment_names:
            if "phi(z)" in name:
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name="{}_phi_z".format(self.name))
            else:
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name="{}_{}".format(self.name, name))
        self.mask_names = ["mask"]
        self.masks = {}
        for name in self.mask_names:
            self.masks[name] = tf.Variable(tf.ones([batch_size,dim]), name="{}_{}".format(self.name, name))

        self.connected_cables = []

    def compile(self):
        info = super().compile()
        info["phi(x)"] = self.act_fx
        return info

    def step(self, injection_table=None, skip_core_calc=False):
        bmask = self.masks.get("mask")
        ########################################################################
        if skip_core_calc == False:
            if self.is_clamped == False:
                # clear any relevant compartments that are NOT stateful before accruing
                # new deposits (this is crucial to ensure any desired stateless properties)
                if self.do_inplace == True:
                    self.compartments["dz"].assign(self.compartments["dz"] * 0)
                else:
                    self.compartments["dz"] = (self.compartments["dz"] * 0)

                if self.do_inplace == True:
                    self.compartments["Jz"].assign(self.compartments["Jz"] * 0)
                else:
                    self.compartments["Jz"] = (self.compartments["Jz"] * 0)

                # gather deposits from any connected nodes & insert them into the
                # right compartments/regions -- deposits in this logic are linearly combined
                for cable in self.connected_cables:
                    deposit = cable.propagate()
                    dest_comp = cable.dest_comp
                    if self.do_inplace == True:
                        self.compartments[dest_comp].assign(self.compartments[dest_comp] + deposit)
                    else:
                        self.compartments[dest_comp] = (deposit + self.compartments[dest_comp])

                # core logic for the (node-internal) dendritic calculation
                dz = self.compartments["dz"]
                '''
                Feedforward integration step
                Equation:
                z <- dz
                '''
                z = dz
                if self.do_inplace == True:
                    self.compartments["z"].assign(z)
                else:
                    self.compartments["z"] = z
            # else, no deposits are accrued (b/c this node is hard-clamped to a signal)
            ########################################################################
            z = self.compartments["z"]

            #c1 = 0.82
            c2 = self.constants["c2"]
            Jz = self.compartments["Jz"]
            #g_tr = self.compartments["g_tr"]
            mask = tf.cast(tf.greater(Jz, 0.0),dtype=tf.float32)
            dS_dJ = transform_utils.sech_sqr(Jz * c2) * mask
            iota = z * dS_dJ
            z = iota

            if self.do_inplace == True:
                self.compartments["z"].assign(z)
            else:
                self.compartments["z"] = (z)

        z = self.compartments["z"]
        # apply post-activation non-linearity
        phi_z = self.fx(z)
        if self.do_inplace == True:
            self.compartments["phi(z)"].assign(phi_z)
        else:
            self.compartments["phi(z)"] = (phi_z)

        if bmask is not None: # applies mask to all component variables of this node
            for key in self.compartments:
                if self.compartments.get(key) is not None:
                    if self.do_inplace == True:
                        self.compartments[key].assign( self.compartments.get(key) * bmask )
                    else:
                        self.compartments[key] = ( self.compartments.get(key) * bmask )
        ########################################################################
        self.t += 1

        # a node returns a list of its named component values
        values = []
        for comp_name in self.compartments:
            comp_value = self.compartments.get(comp_name)
            values.append((self.name, comp_name, comp_value))
        return values
