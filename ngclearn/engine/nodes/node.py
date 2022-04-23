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
from ngclearn.engine.cables.dcable import DCable # dense cable
from ngclearn.engine.cables.scable import SCable # simple cable

class Node:
    """
    Base node element (class from which other node types inherit basic properties from)

    Args:
        node_type: the string concretely denoting this node's type

        dim: number of neurons this node will contain
    """
    def __init__(self, node_type, name, dim):
        self.node_type = node_type
        self.name = name
        self.dim = dim # dimensionality of this node's neural activity
        self.input_nodes = []
        self.input_cables = []

        # node meta-parameters
        self.is_learnable = False
        self.beta = 0.1
        self.leak = 0.0
        self.zeta = 1.0

        # embedded vector statistics
        # self.dz = None
        # self.z = None
        # self.phi_z = None
        self.stat = {}
        self.stat["dz"] = None
        self.stat["z"] = None
        self.stat["phi(z)"] = None
        self.stat["mask"] = None # a binary mask that can be used to make this node's activity values

        self.tick = {}

        self.is_clamped = False

    ############################################################################
    # Setup Routines
    ############################################################################

    def build_tick(self):
        """ Internal function for managing this node's notion of time """
        self.tick = {}
        for key in self.stat:
            self.tick[key] = 0

    def wire_to(self, dest_node, src_var, dest_var, cable_kernel=None, mirror_path_kernel=None, point_to_path=None):
        """
        A wiring function that connects this node to another external node via a cable (or synaptic bundle)

        Args:
            dest_node: destination node (a Node object) to wire this node to

            src_var: name of the compartment inside this node to transmit a signal from (to destination node)

            dest_var: name of the compartment inside the destination node to transmit a signal to

            cable_kernel: Dict defining how to initialize the cable that will connect this node to the destination node.
                The expected keys and corresponding value types are specified below:

                :`'type'`: type of cable to be created.
                    If "dense" is specified, a DCable (dense cable/bundle/matrix of synapses) will be used to
                    transmit/transform information along.

                :`'has_bias'`: if True, adds a bias/base-rate vector to the DCable that be used for wiring

                :`'init'`: type of distribution to use initialize the cable/synapses.
                    If "gaussian" is specified, for example, a Gaussian distribution will be used to initialize
                    each scalar element inside the DCable synaptic matrix (biases are always set to zero)

                :`'seed'`: integer seed to deterministically control initialization of synapses in a DCable

                :Note: either cable_kernel, mirror_path_kernel, or point_to_path MUST be set to something that is not None

            mirror_path_kernel: 2-Tuple that allows a currently existing cable to be re-used as a transformation.
                The value types inside each slot of the tuple are specified below:

                :cable_to_reuse (Tuple[0]): target cable (usually an existing DCable object) to shallow copy and mirror

                :mirror_type (Tuple[1]): how should the cable be mirrored? If "symm_tied" is specified, then the transpose
                    of this cable will be used to transmit information from this node to a destination node, if "anti_symm_tied"
                    is specified, the negative transpose of this cable will be used

                :Note: either cable_kernel, mirror_path_kernel, or point_to_path MUST be set to something that is not None

            point_to_path: a DCable that we want to shallow-copy and directly/identically use to transmit information from this
                node to a destination node (note that its shape/dimensions must be correct, otherwise this will break)

                :Note: either cable_kernel, mirror_path_kernel, or point_to_path MUST be set to something that is not None
        """
        if cable_kernel is None and mirror_path_kernel is None and point_to_path is None:
            print("Error: Must either set |cable_kernel| or |mirror_path_kernel| or |point_to_path| argument! for node({})".format(self.name))
            sys.exit(1)
        cable = None
        if mirror_path_kernel is not None: # directly share/shallow copy this cable but in reverse (with a transpose)
            cable = DCable(inp=(self,src_var),out=(dest_node,dest_var), shared_param_path=mirror_path_kernel, has_bias=False)
            #print(" CREATED:  ",cable.name)
        elif point_to_path is not None: # directly share this cable (a shallow copy)
            has_bias = False
            if cable_kernel is not None:
                has_bias = cable_kernel.get("has_bias")
            cable = DCable(inp=(self,src_var),out=(dest_node,dest_var), point_to=point_to_path, has_bias=has_bias)
        else:
            cable_type = cable_kernel.get("type")
            coeff = cable_kernel.get("coeff")
            if cable_type == "dense":
                if coeff is None:
                    coeff = 1.0
                cable_init = cable_kernel.get("init")
                has_bias = cable_kernel.get("has_bias")
                seed = cable_kernel.get("seed")
                if seed is None:
                    seed = 69
                cable = DCable(inp=(self,src_var),out=(dest_node,dest_var), init_kernel=cable_init, has_bias=has_bias, coeff=coeff, seed=seed)
                #print(" CREATED:  ",cable.name)
            else:
                #coeff = cable_kernel["coeff"]
                cable = SCable(inp=(self,src_var),out=(dest_node,dest_var), coeff=coeff)
        dest_node.input_nodes.append(  self  )
        dest_node.input_cables.append( cable )
        return cable

    def extract(self, var_name):
        """
        Extracts the data signal value that is currently stored inside of a target compartment

        Args:
            var_name: the name of the compartment in this node to extract data from
        """
        return self.stat[var_name]

    def extract_params(self):
        return []

    def inject(self, data):
        """
        Injects an externally provided named value (a vector/matrix) to the desired
        compartment within this node.

        Args:
            data: 2-Tuple containing a named external signal to clamp

                :compartment_name (Tuple[0]): the (str) name of the compartment to clamp this data signal to.

                :signal (Tuple[1]): the data signal block to clamp to the desired compartment name
        """
        var_name, var_value = data
        self.stat[var_name] = var_value

    def clamp(self, data, is_persistent=True):
        """
        Clamps an externally provided named value (a vector/matrix) to the desired
        compartment within this node.

        Args:
            data: 2-Tuple containing a named external signal to clamp

                :compartment_name (Tuple[0]): the (str) name of the compartment to clamp this data signal to.

                :signal (Tuple[1]): the data signal block to clamp to the desired compartment name

            is_persistent: if True, prevents this node from overriding the clamped data over time (Default = True)
        """
        var_name, var_value = data
        self.stat[var_name] = var_value
        if is_persistent is True:
            self.is_clamped = True

    def clear(self):
        """ Wipes/clears values of each compartment in this node (and sets .is_clamped = False). """
        self.build_tick()
        self.is_clamped = False
        self.stat["dz"] = None
        self.stat["z"] = None
        self.stat["phi(z)"] = None
        self.stat["mask"] = None

    def deep_store_state(self):
        """
        Performs a deep copy of all compartment statistics.

        Returns:
            Dict containing a deep copy of each named compartment of this node
        """
        stat_cpy = {}
        for key in self.stat:
            value = self.stat.get(key)
            if value is not None:
                stat_cpy[key] = value + 0
        return stat_cpy

    def check_correctness(self):
        """ Executes a basic wiring correctness check. """
        pass

    ############################################################################
    # Signal Transmission Routines
    ############################################################################

    def step(self, skip_core_calc=False):
        """
        Executes this nodes internal integration/calculation for one discrete step
        in time, i.e., runs simulation of this node for one time step.

        Args:
            skip_core_calc: skips the core components of this node's calculation
        """
        pass

    def calc_update(self, update_radius=-1.0):
        """
        Calculates the updates to local internal synaptic parameters related to this
        specific node given current relevant values (such as node-level precision matrices).

        Args:
            update_radius: radius of Gaussian ball to constrain computed update matrices by
                (i.e., clipping by Frobenius norm)
        """
        return []
