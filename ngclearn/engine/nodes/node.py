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
        self.tick = {}
        for key in self.stat:
            self.tick[key] = 0

    def wire_to(self, dest_node, src_var, dest_var, cable_kernel=None, mirror_path_kernel=None, point_to_path=None):
        if cable_kernel is None and mirror_path_kernel is None and point_to_path is None:
            print("Error: Must either set |cable_kernel| or |mirror_path_kernel| or |point_to_path| argument!")
            sys.exit(1)
        cable = None
        if mirror_path_kernel is not None:
            cable = DCable(inp=(self,src_var),out=(dest_node,dest_var), shared_param_path=mirror_path_kernel, has_bias=False)
            #print(" CREATED:  ",cable.name)
        elif point_to_path is not None:
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
        return self.stat[var_name]

    def extract_params(self):
        return []

    def inject(self, data):
        var_name, var_value = data
        self.stat[var_name] = var_value

    def clamp(self, data, is_persistent=True):
        var_name, var_value = data
        self.stat[var_name] = var_value
        if is_persistent is True:
            self.is_clamped = True

    def clear(self):
        self.build_tick()
        self.is_clamped = False
        self.stat["dz"] = None
        self.stat["z"] = None
        self.stat["phi(z)"] = None
        self.stat["mask"] = None

    def deep_store_state(self):
        stat_cpy = {}
        for key in self.stat:
            value = self.stat.get(key)
            if value is not None:
                stat_cpy[key] = value + 0
        return stat_cpy

    def check_correctness(self):
        pass

    ############################################################################
    # Signal Transmission Routines
    ############################################################################

    def step(self):
        pass

    def calc_update(self, update_radius=-1.0):
        return []
