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
from ngclearn.utils import transform_utils
from ngclearn.engine.cables.cable import Cable

class DCable(Cable):
    """

    Args:
        inp: 2-Tuple defining the nodal points that this cable will connect.
            The value types inside each slot of the tuple are specified below:

            :input_node (Tuple[0]): the source/input Node object that this cable will carry
                signal information from

            :input_compartment (Tuple[1]): the compartment within the source/input Node that
                signals will extracted and transmitted from

        out: 2-Tuple defining the nodal points that this cable will connect.
            The value types inside each slot of the tuple are specified below:

            :input_node (Tuple[0]): the destination/output Node object that this cable will
                carry signal information to

            :input_compartment (Tuple[1]):  the compartment within the destination/output Node that
                signals transmitted and deposited into

        init_kernel:

        has_bias:

        shared_param_path:

        name: the string name of this cable (Default = None which creates an auto-name)

        coeff: a scalar float to control any signal scaling associated with this cable

        seed: integer seed to control determinism of any underlying synapses
            associated with this cable

        point_to:

    """
    def __init__(self, inp, out, init_kernel=None, has_bias=False, shared_param_path=None,
                 name=None, seed=69, coeff=1.0, point_to=None):
        cable_type = "dense"
        super().__init__(cable_type, inp, out, name, seed, coeff=coeff)
        self.coeff = coeff
        self.init_kernel = init_kernel
        self.has_bias = has_bias
        self.shared_param_path = shared_param_path
        self.point_to = point_to
        # special temporal variables for the temp-diff rule variant for self.W
        self.Wmem = None

        if self.shared_param_path is not None:
            cable_to_mirror, path_type = shared_param_path
            """
                path_type = symm_tied, anti_symm_tied
            """
            self.path_type = path_type
            self.W = cable_to_mirror.W
            if self.has_bias is True:
                b = tf.zeros([1,out_dim])
                self.b = tf.Variable(b, name="b_{0}".format(self.name))
        elif self.point_to is not None:
            self.W = self.point_to.W
            self.b = self.point_to.b
        else:
            in_dim = self.inp_node.dim
            out_dim = self.out_node.dim
            init_type = init_kernel[0]
            if init_type == "lkwta":
                if in_dim != out_dim:
                    print("ERROR: input-side dim {0} != output-side dim {1}".format(in_dim,out_dim))
                    sys.exit(1)
                #lat_type = lateral_kernel.get("lat_type")
                n_group = init_kernel[1] #("n_group")
                alpha_scale = init_kernel[2] #("alpha_scale")
                beta_scale = init_kernel[3] #("beta_scale")
                #if n_group > 0:
                # create potential lateral competition synapses within this node
                W = transform_utils.create_competiion_matrix(in_dim, init_type, beta_scale, -alpha_scale, n_group, band=-1)
                self.W = tf.Variable( W, name="W_{0}".format(self.name) )
            else:
                W = transform_utils.init_weights(kernel=self.init_kernel, shape=[in_dim, out_dim], seed=self.seed)
                self.W = tf.Variable(W, name="W_{0}".format(self.name))
            if self.has_bias is True:
                b = tf.zeros([1,out_dim])
                self.b = tf.Variable(b, name="b_{0}".format(self.name))

            self.W_empty = self.W * 0

    #@tf.function
    def propagate(self, node):
        inp_value = node.extract(self.inp_var)
        if self.shared_param_path is not None:
            if self.path_type == "symm_tied":
                out_value = tf.matmul(inp_value, self.W, transpose_b=True)
            elif self.path_type == "anti_symm_tied":
                out_value = tf.matmul(inp_value, -self.W, transpose_b=True)
        else:
            out_value = tf.matmul(inp_value, self.W)

            # if node.node_type == "spike_state":
            #     spike = node.extract("Sz")
            #     #count = np.sign(tf.reduce_sum(spike))
            #     count = tf.reduce_sum(spike) #tf.math.sign(tf.reduce_sum(spike))
            #     #out_value = tf.matmul(inp_value, self.W)
            #     if count > 0.0:
            #         out_value = tf.matmul(inp_value, self.W)
            #     else:
            #         out_value = tf.zeros([inp_value.shape[0], self.out_node.dim])
            # else:
            #     out_value = tf.matmul(inp_value, self.W)

            # if self.name == "z1-to-mu0_dense":
            #     print("***************************")
            #     print("Name:  ",self.name)
            #     print("W.shape = ",self.W.shape)
            #     print("W:\n",self.W)
            #     print(out_value)
            #     print("***************************")
        if self.has_bias is True:
            # print(out_value)
            # print(self.b)
            out_value = out_value + self.b
        out_value = out_value * self.coeff
        #self.cable_out = out_value
        return out_value

    def calc_update(self, update_radius=-1.0):
        if self.is_learnable is True and self.shared_param_path is None:
            # Generalized Hebbian rule (whether or not a node is an error node or not)
            out_signal = self.postact_node.extract(self.postact_comp)
            e_n = out_signal
            if self.deriv_node is not None:
                d_fx = self.deriv_node.dfx( self.deriv_node.extract(self.deriv_comp) )
                e_n = e_n * d_fx

            trigger_update = True # for non-spiking models, this variable is set to True
            if self.preact_node.node_type == "spike_state":
                spike = self.preact_node.extract("Sz")
                count = np.sign(tf.reduce_sum(spike))
                if count > 0.0:
                    trigger_update = True
                else:
                    trigger_update = False

            # apply synaptic update/adjustment if the event is triggered
            if trigger_update is True:
                zf = self.preact_node.extract(self.preact_comp)
                dW = tf.matmul(zf, e_n, transpose_a=True) #* (1.0/Ns)

                if update_radius > 0.0:
                    dW = tf.clip_by_norm(dW, update_radius)
                if self.use_mod_factor is True: # apply modulatory factor matrix to dW
                    W_M = transform_utils.calc_modulatory_factor(self.W)
                    dW = dW * W_M
                dW = -dW * self.gamma
                if self.b is not None:
                    db = tf.reduce_sum(e_n, axis=0, keepdims=True)
                    #if update_radius > 0.0:
                    #    db = tf.clip_by_norm(db, update_radius)
                    db = -db * self.gamma
                    return [dW, db]
                return [dW]
            else:
                # in event-triggered systems (like spiking neurons), it is possible
                # that absolutely no neurons have emitted a binary spike, thus the update is
                # a zero pad tensor the size of the original synaptic tensor
                dW = self.W_empty + 0
                if self.b is not None:
                    db = self.b * 0
                    return [dW, db]
                return [dW]
        return []

    # def clear(self):
    #     self.cable_out = None
