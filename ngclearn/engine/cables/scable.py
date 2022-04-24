import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils
from ngclearn.engine.cables.cable import Cable

class SCable(Cable):
    """
    A simple cable that, at most, applies a scalar amplification of signals that travel across it.
    (Otherwise, this cable works like an identity carry-over.)

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

        coeff: a scalar float to control any signal scaling associated with this cable

        name: the string name of this cable (Default = None which creates an auto-name)

        seed: integer seed to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, inp, out, coeff=1.0, name=None, seed=69):
        cable_type = "simple"
        super().__init__(cable_type, inp, out, name, seed, coeff=coeff)

    def propagate(self, node):
        inp_value = node.extract(self.inp_var)
        out_value = inp_value * self.coeff
        return out_value

    # def clear(self):
    #     self.cable_out = None
