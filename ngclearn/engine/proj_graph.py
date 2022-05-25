import tensorflow as tf
import sys
import numpy as np
import copy

class ProjectionGraph:
    """
    Implements a projection graph -- useful for conducting ancestral sampling of a
    directed generative model or ancestral projection of a clamped graph.
    Note that when instantiating this object, it is important to call .compile(),
    like so:

    | graph = ProjectionGraph(...)
    | info = graph.compile()

    Args:
        name: the name of this projection graph
    """
    def __init__(self, name="sampler"):
        self.name = name
        self.theta = []
        self.exec_cycles = [] # node execution cycles
        self.nodes = {}
        self.cables = {}
        self.learnable_cables = []
        self.learnable_nodes = []
        self.batch_size = 1

    def set_cycle(self, nodes):
        """
        Set execution cycle for this graph

        Args:
            nodes: an ordered list of Node(s) to create an execution cycle for
        """
        self.exec_cycles.append(nodes)
        for j in range(len(nodes)): # collect any learnable cables
            n_j = nodes[j]
            # if n_j.node_type != "feedforward":
            #     print("ERROR: node is of type {0} but must be of type |feedforward_node|".format(n_j.node_type))
            #     sys.exit(1)
            self.nodes[n_j.name] = n_j
            for i in range(len(n_j.connected_cables)):
                cable_i = n_j.connected_cables[i]
                self.cables[cable_i.name] = cable_i
                if cable_i.cable_type == "dense":
                    if cable_i.shared_param_path is None and cable_i.is_learnable is True:
                        self.learnable_cables.append(cable_i)
                        for pname in cable_i.params:
                            param = cable_i.params.get(pname)
                            if param is not None:
                                self.theta.append( param )

    def compile(self, batch_size=-1):
        """
        Executes a global "compile" of this simulation object to ensure internal
        system coherence. (Only call this function after the constructor has been
        set).

        Args:
            batch_size: <UNUSED>

        Returns:
            a dictionary containing post-compilation information about this simulation object
        """
        if batch_size > 0:
            self.batch_size = batch_size
        sim_info = [] # list of hash tables containing properties of each element
                      # in this simulation object
        for i in range(len(self.exec_cycles)):
            cycle_i = self.exec_cycles[i]
            for j in range(len(cycle_i)):
                node_j = cycle_i[j]
                node_j.set_status(status=("dynamic",1))
                info_j = node_j.compile()
                sim_info.append(info_j) # aggregate information hash tables
                #sim_info = {**sim_info, **info_j}
        for cable_name in self.cables:
            info_j = self.cables[cable_name].compile()
            sim_info.append(info_j) # aggregate information hash tables
        return sim_info

    def extract(self, node_name, node_var_name):
        """
        Extract a particular signal from a particular node embedded in this graph

        Args:
            node_name: name of the node from the NGC graph to examine

            node_var_name: compartment name w/in Node to extract signal from

        Returns:
            an extracted signal (vector/matrix) OR None if node does not exist
        """
        if self.nodes.get(node_name) is not None:
            return self.nodes[node_name].extract(node_var_name)
        return None

    def getNode(self, node_name):
        """
        Extract a particular node from this graph

        Args:
            node_name: name of the node from the NGC graph to examine

        Returns:
            the desired Node (object)
        """
        return self.nodes.get(node_name) #self.nodes[node_name]

    def clamp(self, clamp_targets): # inject is a wrapper function over clamp
        """
        Clamps an externally provided named value (a vector/matrix) to the desired
        compartment within a particular Node of this projection graph.

        Args:
            clamp_targets: 3-Tuple containing a named external signal to clamp

                :node_name (Tuple[0]): the (str) name of the node to clamp a data signal to.

                :compartment_name (Tuple[1]): the (str) name of the node's compartment to clamp this data signal to.

                :signal (Tuple[2]): the data signal block to clamp to the desired compartment name

        """
        for clamp_target in clamp_targets:
            node_name, node_comp, node_value = clamp_target
            self.nodes[node_name].clamp((node_comp, node_value))

    def set_cold_state(self, batch_size=-1):
        for i in range(len(self.exec_cycles)):
            cycle_i = self.exec_cycles[i]
            for j in range(len(cycle_i)):
                node_j = cycle_i[j]
                node_j.set_cold_state(batch_size=batch_size)
                #node_j.step(skip_core_calc=True)

    def project(self, clamped_vars=None, readout_vars=None):
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
        if clamped_vars is None:
            clamped_vars = []
        if readout_vars is None:
            readout_vars = []

        if len(clamped_vars) > 0:
            batch_size = clamped_vars[0][2].shape[0]
            self.set_cold_state(batch_size)
        else:
            self.set_cold_state()

        # Step 1: Clamp variables that will persist during sampling/projection step
        for clamped_var in clamped_vars:
            var_name, comp_name, var_value = clamped_var
            if var_value is not None:
                node = self.nodes.get(var_name)
                if node is not None:
                    node.clamp((comp_name, var_value))

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

    def calc_updates(self, debug_map=None):
        """
            Calculates the updates to synaptic weight matrices along each
            learnable wire within this NCN operation graph via a
            generalized Hebbian learning rule.

            Args:
                debug_map: (Default = None), a Dict to place named signals
                    inside (for debugging)
        """
        delta = []
        for j in range(len(self.learnable_cables)):
            cable_j = self.learnable_cables[j]
            delta_j = cable_j.calc_update()
            delta = delta + delta_j
            if debug_map is not None:
                # --------------------------------------------------------------
                # NOTE: this has not been tested...might not work as expected...
                if len(delta_j) == 2: #dW, db
                    if cable_j.params.get("A"):
                        debug_map[cable_j.params["A"].name] = delta_j[0]
                    if cable_j.params.get("b"):
                        debug_map[cable_j.params["b"].name] = delta_j[1]
                else: #dW
                    if cable_j.params.get("A"):
                        debug_map[cable_j.params["A"].name] = delta_j[0]
            # --------------------------------------------------------------
        for j in range(len(self.learnable_nodes)):
            node_j = self.learnable_nodes[j]
            delta_j = node_j.calc_update()
            delta = delta + delta_j
        return delta

    def apply_constraints(self):
        """
        | Apply any constraints to the signals embedded in this graph. This function
            will execute any of the following pre-configured constraints:
        | 1) compute new precision matrices (if applicable)
        | 2) project weights to adhere to vector norm constraints
        """
        # apply constraints to any applicable (learnable) cables
        for j in range(len(self.learnable_cables)):
            cable_j = self.learnable_cables[j]
            cable_j.apply_constraints()
        # apply constraints to any applicable (learnable) nodes
        for j in range(len(self.learnable_nodes)):
            node_j = self.learnable_nodes[j]
            node_j.apply_constraints()

    def clear(self):
        """
        Clears/deletes any persistent signals currently embedded w/in this graph's Nodes
        """
        #self.values = {}
        for node_name in self.nodes:
            node = self.nodes.get(node_name)
            node.clear()
