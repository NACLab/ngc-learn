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

        src_node, src_comp = inp
        self.src_node = src_node
        self.src_comp = src_comp # source compartment
        dest_node, dest_comp = out
        self.dest_node = dest_node
        self.dest_comp = dest_comp # destination compartment

        if name == None:
            self.name = "{0}-to-{1}_{2}".format(src_node.name, dest_node.name, self.cable_type)

        # pre-activity region for first term of local update rule
        self.preact_node = None
        self.preact_comp = None
        # post-activity region for second term of local update rule
        self.postact_node = None
        self.postact_comp = None

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

    def set_update_rule(self, preact, postact, gamma=1.0, use_mod_factor=False):
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
        """
        preact_node, preact_comp = preact
        self.preact_node = preact_node
        self.preact_comp = preact_comp

        postact_node, postact_comp = postact
        self.postact_node = postact_node
        self.postact_comp = postact_comp

        self.gamma = gamma
        self.use_mod_factor = use_mod_factor
        self.is_learnable = True

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
