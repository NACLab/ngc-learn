import tensorflow as tf
import sys
import numpy as np
import copy

class ProjectionGraph:
    """
    Implements a projection graph -- useful for conducting ancestral sampling of a
    directed generative model or ancestral projection of a clamped graph.
    Note that this graph system is NOT learnable.

    Args:
        name: the name of this projection graph

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

        Args:
            nodes: an ordered list of Node(s) to create an execution cycle for
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

        Args:
            node_name: name of the node from the NGC graph to examine

            node_var_name: compartment name w/in Node to extract signal from

        Returns:
            an extracted signal (vector/matrix)
        """
        return self.nodes[node_name].extract(node_var_name)

    def getNode(self, node_name):
        """
        Extract a particular node from this graph

        Args:
            node_name: name of the node from the NGC graph to examine

        Returns:
            the desired Node (object)
        """
        return self.nodes[node_name]

    def project(self, clamped_vars, readout_vars=[]):
        """
        Project signals through the execution pathway(s) defined by this graph

        Args:
            clamped_vars: list of 2-tuples containing named Nodes that will be clamped with particular values.
                Note that this list takes the form: [(node1_name, node_value1), node2_name, node_value2),...]

            readout_vars: list of 2-tuple strings containing named Nodes and their compartments to read signals from.
                Note that this list takes the form: [(node1_name, node1_compartment), node2_name, node2_compartment),...]

        Returns:
            readout values - a list of 3-tuples named signals corresponding to the ones in "readout_vars". Note that
                this list takes the form: [(node1_name, node1_compartment, value), node2_name, node2_compartment, value),...]
        """
        batch_size = 1
        # Step 1: Clamp variables that will persist during sampling/projection step
        for clamped_var in clamped_vars:
            var_name, comp_name, var_value = clamped_var
            batch_size = var_value.shape[0]
            node = self.nodes.get(var_name)
            node.clamp((comp_name, var_value), is_persistent=True) # "z"
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
        Clears/deletes any persistent signals embedded in this graph (clears the state).
        """
        for c in range(len(self.exec_cycles)):
            cycle_c = self.exec_cycles[c]
            for i in range(len(cycle_c)):
                cycle_c[i].clear()
