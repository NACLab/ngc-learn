import tensorflow as tf
import sys
import numpy as np
import copy

"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU LGPL-3.0-or-later license.

You should have received a copy of the XYZ license with
this file. If not, please write to: ago@cs.rit.edu , or visit:
https://www.gnu.org/licenses/lgpl-3.0.en.html
"""

class ProjectionGraph:
    """
    Implements a projection graph -- useful for conducting ancestral sampling of a
    directed generative model or ancestral projection of a clamped graph.
    Note that this graph system is NOT learnable.

    @author: Alexander G. Ororbia
    """
    def __init__(self, name="sampler"):
        self.name = name
        self.theta = []
        self.exec_cycles = [] # node execution cycles
        self.nodes = {}
        self.cables = {}

    def set_cycle(self, nodes):
        """
        Set execution cycle for this graph

        :param nodes:
        :return: None
        """
        self.exec_cycles.append(nodes)
        for j in range(len(nodes)): # collect any learnable cables
            n_j = nodes[j]
            if n_j.node_type != "feedforward":
                print("ERROR: node is of type {0} but must be of type |feedforward_node|".format(n_j.node_type))
                sys.exit(1)
            self.nodes[n_j.name] = n_j
            for i in range(len(n_j.input_cables)):
                cable_i = n_j.input_cables[i]
                self.cables[cable_i.name] = cable_i
                if cable_i.W is not None:
                    self.theta.append(cable_i.W)

    def extract(self, node_name, node_var_name):
        """
        Extract a particular signal from a particular node embedded in this graph

        :param node_name:
        :param node_var_name:
        :return:
        """
        return self.nodes[node_name].extract(node_var_name)

    def getNode(self, node_name):
        """
        Extract a particular node from this graph

        :param node_name:
        :param node_var_name:
        :return:
        """
        return self.nodes[node_name]

    def project(self, clamped_vars, readout_vars=[]):
        """
        Project signals through the execution pathway(s) defined by this graph

        :param clamped_vars:
        :param readout_vars:
        :return: readouts
        """
        batch_size = 1
        # Step 1: Clamp variables that will persist during sampling/projection step
        for clamped_var in clamped_vars:
            var_name, var_value = clamped_var
            batch_size = var_value.shape[0]
            node = self.nodes.get(var_name)
            node.clamp(("z", var_value), is_persistent=True)
            node.step()

        for c in range(len(self.exec_cycles)):
            cycle_c = self.exec_cycles[c]
            for i in range(len(cycle_c)):
                cycle_c[i].step()

        # Post-process NGC graph by extracting predictions at indicated output nodes
        readouts = []
        for var_name, comp_name in readout_vars:
            node = self.nodes.get(var_name)
            var_value = node.extract(comp_name)
            readouts.append( (var_name, comp_name, var_value) )

        return readouts

    def clear(self):
        """
        Clear/delete any persistent signals in this graph
        """
        for c in range(len(self.exec_cycles)):
            cycle_c = self.exec_cycles[c]
            for i in range(len(cycle_c)):
                cycle_c[i].clear()
