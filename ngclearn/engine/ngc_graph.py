import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.engine.cables.cable import Cable
import ngclearn.utils.transform_utils as transform

class NGCGraph:
    """
    Implements the full model structure/graph for an NGC system composed of
    nodes and cables.
    Note that when instantiating this object, it is important to call .compile(),
    like so:

    | graph = NGCGraph(...)
    | info = graph.compile()

    Args:
        K: number of iterative inference/settling steps to simulate

        name: (optional) the name of this projection graph (Default="ncn")

        batch_size: fixed batch-size that the underlying compiled static graph system
            should assume (Note that you can set this also as an argument to .compile() )

            :Note: if "use_graph_optim" is set to False, then this argument is
                not meaningful as the system will work with variable-length batches
    """
    def __init__(self, K=5, name="ncn", batch_size=1):
        self.name = name
        self.theta = [] # set of learnable synaptic parameters
        self.omega = [] # set of non-learnable or slowly-evolved synaptic parameters
        self.exec_cycles = [] # this simulation's execution cycles
        self.nodes = {}
        self.cables = {}
        self.values = {}
        self.values_tmp = []
        self.learnable_cables = []
        self.unique_learnable_objects = {}
        self.learnable_nodes = []
        self.K = K
        self.batch_size = batch_size
        self.use_graph_optim = True
        self.injection_table = {}
        # these data members below are for the .evolve() routine
        self.opt = None
        self.evolve_flag = False

    def set_learning_order(self, param_order):
        """
        Forces this simulation object to arrange its .theta and delta to
        follow a particular order.

        Args:
            param_order: a list of Cables/Nodes which will dictate the strict order
                in which parameter updates will be calculated and how they are
                arranged in .theta (note that delta and theta will match the same
                dictated order)
        """
        self.theta = []
        self.learnable_cables = []
        self.unique_learnable_objects = {}
        self.learnable_nodes = []
        for obj in param_order:
            obj_name = obj.name
            if obj.is_learnable == True:
                if isinstance(obj,Node) == True:
                    self.learnable_nodes.append(n_j)
                    if n_j.node_type == "error": # only error nodes have a possible learnable matrix, i.e., Sigma
                        n_j.compute_precision()
                        self.theta.append(n_j.Sigma)
                    # else, this would break if a novel node is deemed learnable... (FIXME)
                elif isinstance(obj,Cable) == True:
                    self.learnable_cables.append(obj)
                    for pname in obj.params:
                        param = obj.params.get(pname)
                        update_terms = obj.update_terms.get(pname)
                        key_check = "{}.{}".format(obj.name,pname)
                        if update_terms is not None and self.unique_learnable_objects.get(key_check) is None:
                            self.theta.append(param)
                            self.unique_learnable_objects["{}.{}".format(obj.name,pname)] = 1

    def set_cycle(self, nodes, param_order=None):
        """
        Set an execution cycle in this graph

        Args:
            nodes: an ordered list of Node(s) to create an execution cycle for
        """
        cycle = []
        for node in nodes:
            cycle.append(node)
            self.nodes[node.name] = node
            for i in range(len(node.connected_cables)): # for each cable i
                cable_i = node.connected_cables[i]
                self.cables[cable_i.name] = cable_i
                if param_order is None:
                    #if cable_i.cable_type == "dense":
                    #if cable_i.shared_param_path is None and cable_i.is_learnable is True:
                    if cable_i.is_learnable is True:
                        # if cable is learnable (locally), store in theta
                        self.learnable_cables.append(cable_i)
                        for pname in cable_i.params:
                            param = cable_i.params.get(pname)
                            update_terms = cable_i.update_terms.get(pname)
                            key_check = "{}.{}".format(cable_i.name,pname)
                            if update_terms is not None and self.unique_learnable_objects.get(key_check) is None:
                                self.theta.append(param)
                                self.unique_learnable_objects["{}.{}".format(cable_i.name,pname)] = 1
                # else, do NOT set learnable cables b/c the user has specified a *param_order*
        for j in range(len(nodes)): # collect any learnable nodes
            n_j = nodes[j]
            if param_order is None:
                if n_j.is_learnable is True:
                    self.learnable_nodes.append(n_j)
                    if n_j.node_type == "error": # only error nodes have a possible learnable matrix, i.e., Sigma
                        n_j.compute_precision()
                        self.theta.append(n_j.Sigma)
        self.exec_cycles.append(cycle)

        if param_order is not None: # set learnable cables according to order desired by the user
            self.set_learning_order(param_order)

    def set_theta(self, theta_target):
        for j in range(len(self.theta)):
            px_j = theta_target[j]
            self.theta[j].assign(px_j)

    def compile(self, use_graph_optim=True, batch_size=-1):
        """
        Executes a global "compile" of this simulation object to ensure internal
        system coherence. (Only call this function after the constructor has been
        set).

        Args:
            use_graph_optim: if True, this simulation will use static graph
                acceleration (Default = True)

            batch_size: if > 0, will set the integer global batch_size of this
                simulation object (otherwise, self.batch_size will be used)

        Returns:
            a dictionary containing post-compilation information about this simulation object
        """
        if batch_size > 0:
            self.batch_size = batch_size
        if batch_size <= 0:
            batch_size = self.batch_size

        sim_info = [] # list of hash tables containing properties of each element
                      # in this simulation object
        self.use_graph_optim = use_graph_optim
        for i in range(len(self.exec_cycles)):
            cycle_i = self.exec_cycles[i]
            for j in range(len(cycle_i)):
                node_j = cycle_i[j]
                if self.use_graph_optim == True:
                    # all nodes using graph optimization flag MUST be static
                    node_j.set_status(status=("static",self.batch_size))
                else:
                    node_j.set_status(status=("dynamic",self.batch_size))
                info_j = node_j.compile()
                sim_info.append(info_j) # aggregate information hash tables
                #sim_info = {**sim_info, **info_j}
        for cable_name in self.cables:
            info_j = self.cables[cable_name].compile()
            sim_info.append(info_j) # aggregate information hash tables

        if self.use_graph_optim == True:
            # run a resting state settle() to compile the static graph
            for k in range(2): # 3
                self.settle()
            self.clear()
        return sim_info

    def clone_state(self):
        """
        Clones the entire state of this graph (in terms of signals/tensors) and
        stores each node's state dictionary in global has map

        Returns:
            a Dict (hash table) containing string names that map to physical Node objects
        """
        state_map = {} # node name -> node state dictionary
        for i in range(len(self.exec_cycles)):
            cycle_i = self.exec_cycles[i]
            for j in range(len(cycle_i)):
                state_j = cycle_i[j].deep_store_state()
                state_map[cycle_i[j].name] = state_j
        return state_map

    def set_to_state(self, state_map):
        """
        Set every state of every node in this graph to the values contained
        in the global Dict (hash table) "state_map"

        Args:
            state_map: a Dict (hash table) containing string names that map to physical Node objects
        """
        for i in range(len(self.exec_cycles)):
            cycle_i = self.exec_cycles[i]
            for j in range(len(cycle_i)):
                node_j = cycle_i[j]
                state_j = state_map.get(node_j.name)
                for key in node_j.compartments:
                    value_j = state_j.get(key)
                    if value_j is not None:
                        node_j.compartments[key] = value_j + 0

    def extract(self, node_name, node_var_name):
        """
        Extract a particular signal from a particular node embedded in this graph

        Args:
            node_name: name of the node from the NGC graph to examine

            node_var_name: compartment name w/in Node to extract signal from

        Returns:
            an extracted signal (vector/matrix) OR None if either the node does not exist
            or the entire system has not been simulated (meaning that no node dynamics
            have been run yet)
        """
        if len(self.values) == 0:
            return None
        if self.nodes.get(node_name) is not None:
            return self.values.get(node_name).get(node_var_name)
            #return self.nodes[node_name].extract(node_var_name)
        return None

    def getNode(self, node_name):
        """
        Extract a particular node from this graph

        Args:
            node_name: name of the node from the NGC graph to examine

        Returns:
            the desired Node (object) or None if the node does not exist
        """
        return self.nodes.get(node_name) #self.nodes[node_name]

    def clamp(self, clamp_targets): # inject is a wrapper function over clamp
        """
        Clamps an externally provided named value (a vector/matrix) to the desired
        compartment within a particular Node of this NGC graph.
        Note that clamping means this value typically means the value clamped on will persist
        (it will NOT evolve according to the injected node's dynamics over simulation steps,
        unless is_persistent = True).

        Args:
            clamp_targets: 3-Tuple containing a named external signal to clamp

                :node_name (Tuple[0]): the (str) name of the node to clamp a data signal to.

                :compartment_name (Tuple[1]): the (str) name of the node's compartment to clamp this data signal to.

                :signal (Tuple[2]): the data signal block to clamp to the desired compartment name
        """
        for clamp_target in clamp_targets:
            node_name, node_comp, node_value = clamp_target
            self.nodes[node_name].clamp((node_comp, node_value))

            node_inj_table = self.injection_table.get(node_name) # get node table
            if node_inj_table is None:
                node_inj_table = {}
            node_inj_table[node_comp] = 2 # set node's comp to flag 2
            self.injection_table[node_name] = node_inj_table

    def inject(self, injection_targets):
        """
        Injects an externally provided named value (a vector/matrix) to the desired
        compartment within a particular Node of this NGC graph.
        Note that injection means this value does not persist (it will evolve according to the
        injected node's dynamics over simulation steps).

        Args:
            injection_targets: 3-Tuple containing a named external signal to clamp

                :node_name (Tuple[0]): the (str) name of the node to clamp a data signal to.

                :compartment_name (Tuple[1]): the (str) name of the compartment to clamp this data signal to.

                :signal (Tuple[2]): the data signal block to clamp to the desired compartment name

            is_persistent: if True, clamped data value will persist throughout simulation (Default = True)
        """
        for clamp_target in injection_targets:
            node_name, node_comp, node_value = clamp_target
            node = self.getNode(node_name)
            node.inject((node_comp, node_value))

            node_inj_table = self.injection_table.get(node_name) # get node table
            if node_inj_table is None:
                node_inj_table = {}
            node_inj_table[node_comp] = 1 # set node's comp to flag 1
            self.injection_table[node_name] = node_inj_table

    def set_to_resting_state(self, batch_size=-1):
        # Initialize the values of every non-clamped node
        for i in range(len(self.exec_cycles)):
            cycle_i = self.exec_cycles[i]
            for j in range(len(cycle_i)):
                node_j = cycle_i[j]
                node_inj_table = self.injection_table.get(node_j.name)
                if node_inj_table is None:
                    node_inj_table = {}
                if batch_size > 0:
                    node_j.set_cold_state(node_inj_table, batch_size=batch_size)
                node_j.step(node_inj_table, skip_core_calc=True)

    def parse_node_values(self, node_values):
        for ii in range(len(node_values)):
            node_name, comp_name, comp_value = node_values[ii]
            if self.use_graph_optim == True:
                node_name = node_name.numpy().decode('ascii')
                comp_name = comp_name.numpy().decode('ascii')
            vdict = self.values.get(node_name)
            if vdict is not None:
                vdict[comp_name] = comp_value
                self.values[node_name] = vdict
            else:
                vdict = {}
                vdict[comp_name] = comp_value
                self.values[node_name] = vdict

    # TODO: add in early-stopping to settle routine...
    def settle(self, clamped_vars=None, readout_vars=None, init_vars=None, cold_start=True, K=-1,
               debug=False, masked_vars=None, calc_delta=True):
        """
        Execute this NGC graph's iterative inference using the execution pathway(s)
        defined at construction/initialization.

        Args:
            clamped_vars: list of 3-tuple strings containing named Nodes, their compartments, and values to (persistently)
                clamp on. Note that this list takes the form:
                [(node1_name, node1_compartment, value), node2_name, node2_compartment, value),...]

            readout_vars: list of 2-tuple strings containing Nodes and their compartments to read from (in this function's output).
                Note that this list takes the form:
                [(node1_name, node1_compartment), node2_name, node2_compartment),...]

            init_vars: list of 3-tuple strings containing named Nodes, their compartments, and values to initialize each
                Node from. Note that this list takes the form:
                [(node1_name, node1_compartment, value), node2_name, node2_compartment, value),...]

            cold_start: initialize all non-clamped/initialized Nodes (i.e., their compartments contain None)
                to zero-vector starting points/resting states

            K: number simulation steps to run (Default = -1), if <= 0, then self.K will
                be used instead

            debug: <UNUSED>

            masked_vars: list of 4-tuple that instruct which nodes/compartments/masks/clamped values to apply.
                This list is used to trigger auto-associative recalls from this NGC graph.
                Note that this list takes the form:
                [(node1_name, node1_compartment, mask, value), node2_name, node2_compartment, mask, value),...]

            calc_delta: compute the list of synaptic updates for each learnable
                parameter within .theta? (Default = True)

        Returns:
            readouts, delta;
                where "readouts" is a 3-tuple list of the form [(node1_name, node1_compartment, value),
                node2_name, node2_compartment, value),...], and
                "delta" is a list of synaptic adjustment matrices (in the same order as .theta)
        """
        if clamped_vars is None:
            clamped_vars = []
        if readout_vars is None:
            readout_vars = []
        if init_vars is None:
            init_vars = []
        if masked_vars is None:
            masked_vars = []
        sim_batch_size = -1

        K_ = K
        if K_ < 0:
            K_ = self.K

        # Case 1: Clamp variables that will persist during settling/inference
        self.clamp(clamped_vars)
        if len(clamped_vars) > 0:
            for ii in range(len(clamped_vars)):
                data = clamped_vars[ii][2]
                _batch_size = data.shape[0]
                if sim_batch_size > 0 and _batch_size != sim_batch_size:
                    print("ERROR: clamped_vars - cannot provide mixed batch lengths: " \
                          "item {} w/ shape[0] {} != sim_batch_size of {}".format(
                          ii, _batch_size, sim_batch_size))
                sim_batch_size = _batch_size

        # Case 2: Clamp variables that will NOT persist during settling/inference
        self.inject(init_vars)
        if len(init_vars) > 0:
            for ii in range(len(clamped_vars)):
                data = init_vars[ii][2]
                _batch_size = data.shape[0]
                if sim_batch_size > 0 and _batch_size != sim_batch_size:
                    print("ERROR: inject_vars - cannot provide mixed batch lengths: " \
                          "item {} w/ shape[0] {} != sim_batch_size of {}".format(
                          ii, _batch_size, sim_batch_size))
                sim_batch_size = _batch_size

        if cold_start is True:
            self.set_to_resting_state(batch_size=sim_batch_size)
        # TODO: re-integrate back this block of code
        # apply any desired masking variables
        # for masked_var in masked_vars:
        #     var_name, var_comp, mask, clamped_val = masked_var
        #     node = self.nodes.get(var_name)
        #     if node is not None:
        #         curr_val = node.extract(var_comp) #("z")
        #         curr_val = clamped_val * mask + curr_val * (1.0 - mask)
        #         node.clamp( (var_comp, curr_val), is_persistent=False)
        #         node.step(skip_core_calc=True)
        #     else:
        #         print("Node({}) does not exist for masking target".format(var_name))

        delta = None
        node_values = None
        for k in range(K_):
            if calc_delta == True:
                if k == K_-1:
                    node_values, delta = self._run_step(calc_delta=True, use_optim=self.use_graph_optim)
                else: # delta is computed at very last iteration of the simulation
                    node_values, delta = self._run_step(calc_delta=False, use_optim=self.use_graph_optim)
            else: # OR, never compute delta inside the simulation
                node_values, delta = self._run_step(calc_delta=False, use_optim=self.use_graph_optim)
            # TODO: move back in masking code here (or inside static graph...)

        # parse results from static graph & place correct shallow-copied items in system dictionary
        self.parse_node_values(node_values)
        #########################################################################
        # Post-process NGC graph by extracting predictions at indicated output nodes
        #########################################################################
        readouts = []
        for var_name, comp_name in readout_vars:
            value = self.values.get(var_name).get(comp_name)
            readouts.append( (var_name, comp_name, value) )
        return readouts, delta

    def step(self, calc_delta=False):
        """
        Online function for simulating exactly one discrete time step of this
        simulated NGC graph given its exact current state.

        Args:
            calc_delta: compute the list of synaptic updates for each learnable
                parameter within .theta? (Default = True)

        Returns:
            readouts, delta;
                where "readouts" is a 3-tuple list of the form [(node1_name, node1_compartment, value),
                node2_name, node2_compartment, value),...], and
                "delta" is a list of synaptic adjustment matrices (in the same order as .theta)
        """
        values, delta = self._run_step(calc_delta=calc_delta, use_optim=self.use_graph_optim)
        self.parse_node_values(values)
        return delta

    def _run_step(self, calc_delta=False, use_optim=False):
        """ Internal function to run step (do not call externally!)"""
        # feed in injection table externally to node-system (to avoid getting
        # compiled as part of the static graph)
        if use_optim == True:
            values, delta = self._step_fast(self.injection_table, calc_delta)
        else:
            values, delta = self._step(self.injection_table, calc_delta)
        # update injection look-up table
        for node_name in self.injection_table:
            node_table = self.injection_table.get(node_name)
            for comp_name in node_table:
                inj_value = node_table.get(comp_name)
                if inj_value == 1:
                    node_table[comp_name] = None
        return values, delta

    @tf.function
    def _step_fast(self, injection_table, calc_delta=False): # optimized call to _step()
        values, delta = self._step(injection_table, calc_delta)
        return values, delta

    def _step(self, injection_table, calc_delta=False):
        delta = None
        values = []
        for cycle in self.exec_cycles:
            for node in cycle:
                node_inj_table = injection_table.get(node.name)
                if node_inj_table is None:
                    node_inj_table = {}
                node_values = node.step(node_inj_table)
                values = values + node_values
        if calc_delta == True:
            delta = self.calc_updates()
        if self.opt is not None and self.evolve_flag == True:
            if delta is not None:
                self.opt.apply_gradients(zip(delta, self.theta))
                self.apply_constraints()
                self.clear()
        return values, delta

    def calc_updates(self, debug_map=None):
        """
            Calculates the updates to synaptic weight matrices along each
            learnable wire within this graph via a
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
            Prec_l = node_j.apply_constraints()

    def set_optimization(self, opt_algo):
        """
        Sets the internal optimization algorithm used by this simulation object.

        Args:
            opt_algo: optimization algorithm to be used, e.g., SGD, Adam, etc.
                (Note: must be a valid TF2 optimizer.)
        """
        self.opt = opt_algo

    def evolve(self, clamped_vars=None, readout_vars=None, init_vars=None,
               cold_start=True, K=-1, masked_vars=None):
        """
        Evolves this simulation object for one full K-step episode given
        input information through clamped and initialized variables. Note that
        this is a *convenience function* written to embody an NGC system's
        full settling process, its local synaptic update calculations, as well
        as the optimization of and application of constraints to the synaptic
        parameters contained within .theta.

        Args:
            clamped_vars: list of 3-tuple strings containing named Nodes, their compartments, and values to (persistently)
                clamp on. Note that this list takes the form:
                [(node1_name, node1_compartment, value), node2_name, node2_compartment, value),...]

            readout_vars: list of 2-tuple strings containing Nodes and their compartments to read from (in this function's output).
                Note that this list takes the form:
                [(node1_name, node1_compartment), node2_name, node2_compartment),...]

            init_vars: list of 3-tuple strings containing named Nodes, their compartments, and values to initialize each
                Node from. Note that this list takes the form:
                [(node1_name, node1_compartment, value), node2_name, node2_compartment, value),...]

            cold_start: initialize all non-clamped/initialized Nodes (i.e., their compartments contain None)
                to zero-vector starting points

            K: number simulation steps to run (Default = -1), if <= 0, then self.K will
                be used instead

            masked_vars: list of 4-tuple that instruct which nodes/compartments/masks/clamped values to apply.
                This list is used to trigger auto-associative recalls from this NGC graph.
                Note that this list takes the form:
                [(node1_name, node1_compartment, mask, value), node2_name, node2_compartment, mask, value),...]

        Returns:
            readouts, delta;
                where "readouts" is a 3-tuple list of the form [(node1_name, node1_compartment, value),
                node2_name, node2_compartment, value),...]
        """
        self.evolve_flag = True
        readouts, _ = self.settle(
                        clamped_vars=clamped_vars, readout_vars=readout_vars,
                        init_vars=init_vars, cold_start=cold_start, K=K,
                        masked_vars=masked_vars, calc_delta=True
                      )
        return readouts

    def clear(self, batch_size=-1):
        """
        Clears/deletes any persistent signals currently embedded w/in this graph's Nodes
        """
        self.values = {}
        self.values_tmp = []
        self.injection_table = {}
        for node_name in self.nodes:
            node = self.nodes.get(node_name)
            node.clear(batch_size=batch_size)
