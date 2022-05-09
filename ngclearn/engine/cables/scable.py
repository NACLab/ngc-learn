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
        super().__init__(cable_type, inp, out, name, seed)
        self.coeff = coeff

    def compile(self):
        """
        Executes the "compile()" routine for this node. Sub-class nodes can
        extend this in case they contain other elements besides compartments
        that must be configured properly for global simulation usage.

        Returns:
            a dictionary containing post-compilation check information about this cable
        """
        info = super().compile()
        info["coefficient"] = self.coeff
        return info

    def propagate(self):
        in_signal = self.src_node.extract(self.src_comp) # extract input signal
        out_signal = in_signal * self.coeff
        return out_signal
