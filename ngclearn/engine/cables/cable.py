import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils

class Cable:
    """
    Base cable element (class from which other cable types inherit basic properties from)

    Args:
        cable_type: the string concretely denoting this cable's type

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

        name: the string name of this cable (Default = None which creates an auto-name)

        seed: integer seed to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, cable_type, inp, out, name=None, seed=69):
        self.cable_type = cable_type
        self.seed = seed
        self.name = name
        self.is_learnable = False
        self.constraint_kernel = None
        self.decay_kernel = None

        src_node, src_comp = inp
        self.src_node = src_node
        self.src_comp = src_comp # source compartment
        dest_node, dest_comp = out
        self.dest_node = dest_node
        self.dest_comp = dest_comp # destination compartment

        if name == None:
            self.name = "{0}-to-{1}_{2}".format(src_node.name, dest_node.name, self.cable_type)
        self.short_name = self.name

        self.params = {} # what internal synaptic/learnable parameters belong to this cable
        self.update_terms = {} # maps param names to their ordered tuples of update terms

    def compile(self):
        """
        Executes the "compile()" routine for this cable. Sub-class cables can
        extend this in case they contain other elements that must be configured
        properly for global simulation usage.

        Returns:
            a dictionary containing post-compilation check information about this cable
        """
        info = {} # hash table to store any useful information for post-compile analysis
        if self.src_node is None:
            print("Source node missing for cable {}".format(self.name))
        if self.src_comp is None:
            print("Source compartment missing for cable {}".format(self.name))
        if self.dest_node is None:
            print("Destination node missing for cable {}".format(self.name))
        if self.dest_comp is None:
            print("Destination compartment missing for cable {}".format(self.name))
        info["object_type"] = self.cable_type
        info["object_name"] = self.name
        info["is_learnable"] = self.is_learnable
        info["seed"] = self.seed
        info["src_node"] = self.src_node.name
        info["src_comp"] = self.src_comp
        info["dest_node"] = self.src_node.name
        info["dest_comp"] = self.src_comp
        return info

    def set_constraint(self, constraint_kernel):
        self.constraint_kernel = constraint_kernel

    def set_decay(self, decay_kernel):
        self.decay_kernel = decay_kernel

    def set_update_rule(self, preact=None, postact=None, gamma=1.0, use_mod_factor=False,
                        param=None, decay_kernel=None):
        """
        Sets the synaptic adjustment rule for this cable (currently a 2-factor local synaptic Hebbian update rule).

        Args:
            preact: 2-Tuple defining the pre-activity/source node of which the first factor the synaptic
                update rule will be extracted from.
                The value types inside each slot of the tuple are specified below:

                :preact_node (Tuple[0]): the physical node that offers a pre-activity signal for the first
                    factor of the synaptic/cable update

                :preact_compartment (Tuple[1]): the component in the preact_node to extract the necessary
                    signal to compute the first factor the synaptic/cable update

            postact: 2-Tuple defining the post-activity/source node of which the second factor the synaptic
                update rule will be extracted from.
                The value types inside each slot of the tuple are specified below:

                :postact_node (Tuple[0]): the physical node that offers a post-activity signal for the second
                    factor of the synaptic/cable update

                :postact_compartment (Tuple[1]): the component in the postact_node to extract the necessary
                    signal to compute the second factor the synaptic/cable update

            gamma: scaling factor for the synaptic update

            use_mod_factor: if True, triggers the modulatory matrix weighting factor to be
                applied to the resultant synaptic update

                :Note: This is un-tested/not fully integrated

            param: a list of strings, each containing named parameters that are to be learned w/in this cable

            decay_kernel: 2-Tuple defining the type of weight decay to be applied to the synapses.
                The value types inside each slot of the tuple are specified below:

                :decay_type (Tuple[0]): string indicating which type of weight decay to use,
                    "l2" will trigger L2-penalty decay, while "l1" will trigger L1-penalty decay

                :decay_coefficient (Tuple[1]): scalar/float to control magnitude of decay applied
                    to computed local updates
        """
        pass

    def propagate(self):
        """
        Internal transmission function that computes the correct transformation
        of a source node to a destination node

        Returns:
            the resultant transformed signal (transformation f information from "node")
        """
        return 0.0

    def calc_update(self):
        """
        Calculates the updates to the internal synapses that compose this cable
        given this cable's pre-configured synaptic update rule.

        Args:
            clip_kernel: radius of Gaussian ball to constrain computed update matrices by
                (i.e., clipping by Frobenius norm)
        """
        return []

    def apply_constraints(self):
        """
        Apply any constraints to the learnable parameters contained within
        this cable.
        """
        pass
