import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils

class FNode(Node):
    """
    | Implements a feedforward (stateless) transmission node:
    |   z = dz
    | where:
    |   dz - aggregated input signals from other nodes/locations

    | Compartments:
    |   * dz - incoming pressures/signals (deposited signals summed)
    |   * z - the state values/neural activities, set as: z = dz
    |   * phi(z) -  the post-activation of the neural activities

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        act_fx: activation function -- phi(v) -- to apply to neural activities
    """
    def __init__(self, name, dim, act_fx="identity"):
        node_type = "feedforward"
        super().__init__(node_type, name, dim)

        fx, dfx = transform_utils.decide_fun(act_fx)
        self.fx = fx
        self.dfx = dfx

        self.build_tick()

    def check_correctness(self):
        """ Executes a basic wiring correctness check. """
        is_correct = True
        for j in range(len(self.input_nodes)):
            n_j = self.input_nodes[j]
            cable_j = self.input_cables[j]
            dest_var_j = cable_j.out_var
            if dest_var_j != "dz":
                is_correct = False
                print("ERROR: Cable {0} mis-wires to {1}.{2} (can only be .dz)".format(cable_j.name, self.name, dest_var_j))
                break
        return is_correct

    ############################################################################
    # Signal Transmission Routines
    ############################################################################

    def step(self, skip_core_calc=False):
        z = self.stat.get("z")
        phi_z = self.stat["phi(z)"]
        if self.is_clamped is False and skip_core_calc is False:

            for j in range(len(self.input_nodes)):
                n_j = self.input_nodes[j]
                cable_j = self.input_cables[j]
                dest_var_j = cable_j.out_var
                # print("Parent ",n_j.name)
                # print("     z = ",n_j.extract("z"))
                # print("phi(z) = ",n_j.extract("phi(z)"))
                tick_j = self.tick.get(dest_var_j)
                var_j = self.stat.get(dest_var_j) # get current value of component
                dz_j = cable_j.propagate(n_j)
                if tick_j > 0: #if var_j is not None:
                    var_j = var_j + dz_j
                else:
                    var_j = dz_j
                self.stat[dest_var_j] = var_j
                self.tick[dest_var_j] = self.tick[dest_var_j] + 1
            dz = self.stat.get("dz")
            if dz is None:
                dz = 0.0

            """
            Feedforward integration step
            Equation:
            z <- dz
            """
            z = dz
        # the post-activation function is computed always, even if pre-activation is clamped
        phi_z = self.fx(z)
        self.stat["z"] = z
        self.stat["phi(z)"] = phi_z
        self.build_tick()
