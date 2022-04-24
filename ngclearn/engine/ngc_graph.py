import tensorflow as tf
import sys
import numpy as np
import copy

class NGCGraph:
    """
    Implements the full model structure/graph for an NGC system composed of
    nodes and cables.

    Args:
        K: number of iterative inference/settling steps to simulate

        name: (optional) the name of this projection graph (Default="ncn")

    @author: Alexander G. Ororbia
    """
    def __init__(self, K=10, name="ncn"):
        self.name = name
        self.theta = [] # set of learnable synaptic parameters
        self.omega = [] # set of non-learnable or slowly-evolved synaptic parameters
        self.exec_cycles = [] # node execution cycles
        self.nodes = {}
        self.cables = {}
        self.learnable_cables = []
        self.learnable_nodes = []

        self.evolved_cable_pairs = [] # (target_cable_name_to_alter, src_cable_name)

        # inference meta-parameters
        self.K = K
        # learning meta-parameters
        self.proj_update_mag = 1.0
        self.proj_weight_mag = 1.0 #2.0
        self.param_axis = 0

    def set_cycle(self, nodes):
        """
        Set an execution cycle in this graph

        Args:
            nodes: an ordered list of Node(s) to create an execution cycle for
        """
        self.exec_cycles.append(nodes)
        for j in range(len(nodes)): # collect any learnable cables
            n_j = nodes[j]
            self.nodes[n_j.name] = n_j
            for i in range(len(n_j.input_cables)): # for each cable i
                cable_i = n_j.input_cables[i]
                self.cables[cable_i.name] = cable_i
                if cable_i.cable_type == "dense":
                    if cable_i.shared_param_path is None and cable_i.is_learnable is True:
                        # if cable is learnable (locally), store in theta
                        self.learnable_cables.append(cable_i)
                        if cable_i.W is not None:
                            self.theta.append(cable_i.W)
                        if cable_i.b is not None:
                            self.theta.append(cable_i.b)
                    # else:
                    #     # if cable is non-learnable or slowly evolved, store in omega
                    #     if cable_i.W is not None:
                    #         self.omega.append(cable_i.W)
                    #     if cable_i.b is not None:
                    #         self.omega.append(cable_i.b)

        for j in range(len(nodes)): # collect any learnable nodes
            n_j = nodes[j]
            if n_j.is_learnable is True:
                self.learnable_nodes.append(n_j)
                if n_j.node_type == "error": # only error nodes have a possible learnable matrix, i.e., Sigma
                    n_j.compute_precision()
                    self.theta.append(n_j.Sigma)

    def set_evolve_pair(self, pair):
        """
        NOTE: Untested function - not yet integrated fully
        """
        targName, srcName = pair
        self.evolved_cable_pairs.append( (targName, srcName) )
        targ_cable = self.cables.get(targName)
        if targ_cable.W is not None:
            self.omega.append(targ_cable.W)
        if targ_cable.b is not None:
            self.omega.append(targ_cable.b)

    def check_correctness(self):
        """
        Executes a basic correctness check for this graph (examines if the structure is valid)
        """
        flag = True
        for i in range(len(self.exec_cycles)):
            cycle_i = self.exec_cycles[i]
            for j in range(len(cycle_i)):
                node_j = cycle_i[j]
                flag = node_j.check_correctness()
                if flag is False:
                    break
        if flag is False:
            print("ERROR: NGC-graph is incorrect!")
        return flag

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
                for key in node_j.stat:
                    value_j = state_j.get(key)
                    if value_j is not None:
                        node_j.stat[key] = value_j + 0

    def extract(self, node_name, node_var_name):
        """
        Extract a particular signal from a particular node embedded in this graph

        Args:
            node_name: name of the node from the NGC graph to examine

            node_var_name: compartment name w/in Node to extract signal from

        Returns:
            an extracted signal (vector/matrix)
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

    def clamp(self, node_target, data, is_persistent=True):
        """
        Clamps an externally provided named value (a vector/matrix) to the desired
        compartment within a particular Node of this NGC graph.
        Note that clamping means this value typically means the value clamped on will persist
        (it will NOT evolve according to the injected node's dynamics over simulation steps,
        unless is_persistent = True).

        Args:
            node_name: 2-Tuple containing a named external signal to clamp

                :node_name (Tuple[0]): the (str) name of the node to clamp this data signal to.

                :ompartment_name (Tuple[1]): the (str) name of the compartment to clamp this data signal to.

            signal: the data signal block to clamp to the desired compartment name

            is_persistent: if True, clamped data value will persist throughout simulation (Default = True)
        """
        node = self.getNode(node_target)
        var_name, var_value = data
        node.clamp((var_name, var_value), is_persistent=is_persistent)

    def inject(self, node_name, data):
        """
        Injects an externally provided named value (a vector/matrix) to the desired
        compartment within a particular Node of this NGC graph.
        Note that injection means this value does not persist (it will evolve according to the
        injected node's dynamics over simulation steps).

        Args:
            node_name: 2-Tuple containing a named external signal to clamp

                :node_name (Tuple[0]): the (str) name of the node to clamp this data signal to.

                :ompartment_name (Tuple[1]): the (str) name of the compartment to clamp this data signal to.

            signal: the data signal block to clamp to the desired compartment name
        """
        node = self.getNode(node_name)
        var_name, var_value = data
        node.inject((var_name, var_value))

    #@tf.function
    def settle(self, clamped_vars=[], readout_vars=[], init_vars=[], cold_start=True, K=-1,
               debug=False, masked_vars=[]):
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
                to zero-vector starting points

            K: number simulation steps to run (Default = -1), if <= 0, then self.K will
                be used instead

            debug: <UNUSED>

            masked_vars: list of 4-tuple that instruct which nodes/compartments/masks/clamped values to apply.
                This list is used to trigger auto-associative recalls from this NGC graph.
                Note that this list takes the form:
                [(node1_name, node1_compartment, mask, value), node2_name, node2_compartment, mask, value),...]

        Returns:
            readouts - a 3-tuple list of the form [(node1_name, node1_compartment, value), node2_name, node2_compartment, value),...]
        """
        #########################################################################
        # Model graph/system setup
        #########################################################################
        batch_size = 1
        # Case 1: Clamp variables that will persist during settling/inference
        for clamped_var in clamped_vars:
            var_name, comp_name, var_value = clamped_var
            if var_value is not None:
                batch_size = var_value.shape[0]
                node = self.nodes.get(var_name)
                if node is not None:
                    node.clamp((comp_name, var_value), is_persistent=True) #"z"
                    node.step()
            # else, CANNOT clamp a variable value to None

        # Case 2: Clamp variables that will NOT persist during settling/inference
        for clamped_var in init_vars:
            var_name, comp_name, var_value = clamped_var
            if var_value is not None:
                batch_size = var_value.shape[0]
                node = self.nodes.get(var_name)
                if node is not None:
                    node.clamp( (comp_name, var_value), is_persistent=False) #"z"
                    node.step(skip_core_calc=True)
            # else, CANNOT init a variable value with None

        if cold_start is True:
            # Initialize the state values of every non-clamped node to zero matrices
            for i in range(len(self.exec_cycles)):
                cycle_i = self.exec_cycles[i]
                for j in range(len(cycle_i)):
                    node_j = cycle_i[j]
                    if node_j.node_type == "spike_state": # spiking neurons take extra care to zero-init
                        pad = tf.zeros([batch_size, node_j.dim])
                        node_j.clamp( ("Jz", pad), is_persistent=False)
                        node_j.clamp( ("Vz", pad+0), is_persistent=False)
                        node_j.clamp( ("rfr_z", pad+0), is_persistent=False)
                        node_j.clamp( ("Sz", pad+0), is_persistent=False)
                        node_j.clamp( ("phi(z)", pad+0), is_persistent=False)
                        node_j.step(skip_core_calc=True)
                    else:
                        if node_j.extract("z") is None:
                            node_j.clamp( ("z", tf.zeros([batch_size, node_j.dim])), is_persistent=False)
                            node_j.step(skip_core_calc=True)

            # apply any desired masking variables
            for masked_var in masked_vars:
                var_name, var_comp, mask, clamped_val = masked_var
                node = self.nodes.get(var_name)
                if node is not None:
                    curr_val = node.extract(var_comp) #("z")
                    curr_val = clamped_val * mask + curr_val * (1.0 - mask)
                    node.clamp( (var_comp, curr_val), is_persistent=False)
                    node.step(skip_core_calc=True)
                else:
                    print("Node({}) does not exist for masking target".format(var_name))

        def debug_print():
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("State Neurons:")
            print("z0.zf: ",self.extract("z0","phi(z)").numpy())
            print("z1.zf: ",self.extract("z1","phi(z)").numpy())
            print("z2.zf: ",self.extract("z2","phi(z)").numpy())
            print("z3.zf: ",self.extract("z3","phi(z)").numpy())
            print("Expectations:")
            print("mu0.zf: ",self.extract("mu0","phi(z)").numpy())
            print("mu1.zf: ",self.extract("mu1","phi(z)").numpy())
            print("mu2.zf: ",self.extract("mu2","phi(z)").numpy())
            print("Error Neurons:")
            print("e0.zf: ",self.extract("e0","phi(z)").numpy())
            print("e1.zf: ",self.extract("e1","phi(z)").numpy())
            print("e2.zf: ",self.extract("e2","phi(z)").numpy())

            print("Stats:")
            print("e1_tm1.zf: ",self.extract("e1_tm1","phi(z)").numpy())
            print("e2_tm1.zf: ",self.extract("e2_tm1","phi(z)").numpy())
            print("de1.zf: ",self.extract("de1","phi(z)").numpy())
            print("de2.zf: ",self.extract("de2","phi(z)").numpy())

            #print("z0.z : ",self.extract("z0","z").numpy())
            # print("z0.zf: ",self.extract("z0","phi(z)").numpy())
            # #print("e0.z : ",self.extract("e0","z").numpy())
            # print("e0.zf: ",self.extract("e0","phi(z)").numpy())
            # #print("mu0.z : ",self.extract("mu0","z").numpy())
            # #print("mu0.zf: ",self.extract("mu0","phi(z)").numpy())
            # #print("e1.z : ",self.extract("e1","z").numpy())
            # print("z1.zf: ",self.extract("z1","phi(z)").numpy())
            # print("e1.zf: ",self.extract("e1","phi(z)").numpy())
            # #print("z1.z : ",self.extract("z1","z").numpy())
            #
            # #print("mu1.z : ",self.extract("mu1","z").numpy())
            # #print("mu1.zf: ",self.extract("mu1","phi(z)").numpy())
            # #print("z2.z : ",self.extract("z2","z").numpy())
            # print("z2.zf: ",self.extract("z2","phi(z)").numpy())

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        K_ = K
        if K_ < 0:
            K_ = self.K
        # if debug is True:
        #     print("STEP -1")
        #     debug_print()

        #########################################################################
        # Conduct inference by running through execution cycles
        #########################################################################
        for k in range(K_):
            ####################################################################
            # Every step inside this for-loop is one iteration of NGC inference
            for i in range(len(self.exec_cycles)): # for each programmed exec cycle
                cycle_i = self.exec_cycles[i]
                for j in range(len(cycle_i)): # for each node (in order) in exec cycle i
                    node_j = cycle_i[j]
                    node_j.step()
                    # print("*****************")
                    # print(node_j.name)
                    # print(node_j.extract("z"))
                    # print(node_j.extract("phi(z)"))

                # apply any masking variables at this stage in inference
                for masked_var in masked_vars:
                    var_name, var_comp, mask, clamped_val = masked_var
                    node = self.nodes.get(var_name)
                    if node is not None:
                        curr_val = node.extract(var_comp) #("z")
                        curr_val = clamped_val * mask + curr_val * (1.0 - mask)
                        node.clamp( (var_comp, curr_val), is_persistent=False)
                        node.step(skip_core_calc=True)
                    else:
                        print("Node({}) does not exist for masking target".format(var_name))
            ####################################################################
            if debug is True:
                print("___STEP ",k)
                debug_print()
        #########################################################################

        #########################################################################
        # Post-process NGC graph by extracting predictions at indicated output nodes
        #########################################################################
        readouts = []
        for var_name, comp_name in readout_vars:
            node = self.nodes.get(var_name)
            var_value = node.extract(comp_name)
            readouts.append( (var_name, comp_name, var_value) )

        return readouts

    def calc_updates(self, debug_map=None):
        """
            Calculates the updates to synaptic weight matrices along each
            non-identity wire within this NCN operation graph via a
            generalized Hebbian learning rule (or via calculus if a wire
            is an attention wire).

            Args:
                debug_map: (Default = None), a Dict to place named signals inside (for debugging)
        """
        delta = []
        for j in range(len(self.learnable_cables)):
            cable_j = self.learnable_cables[j]
            delta_j = cable_j.calc_update(update_radius=self.proj_update_mag)
            delta = delta + delta_j
            if debug_map is not None:
                if len(delta_j) == 2: #dW, db
                    debug_map[cable_j.W.name] = delta_j[0]
                    debug_map[cable_j.b.name] = delta_j[1]
                else: #dW
                    debug_map[cable_j.W.name] = delta_j[0]
        for j in range(len(self.learnable_nodes)):
            node_j = self.learnable_nodes[j]
            delta_j = node_j.calc_update()
            delta = delta + delta_j
        return delta

    def apply_constraints(self):
        """
        | Apply any constraints to the signals embedded in this graph. This function
            will execute any of the following pre-configured constraints:
        | 1) compute new precision matrices
        | 2) project weights to adhere to vector norm constraints
        """
        # compute error node precision synapses
        for j in range(len(self.learnable_nodes)):
            node_j = self.learnable_nodes[j]
            node_j.compute_precision()
            # print(node_j.name)
            # print("Sigma:\n",node_j.Sigma)
            # print("Prec :\n",node_j.Prec)
        # apply constraints
        if self.proj_weight_mag > 0.0:
            for j in range(len(self.learnable_cables)):
                cable_j = self.learnable_cables[j]
                #print("W.before:\n",tf.norm(cable_j.W,axis=self.param_axis))
                cable_j.W.assign(tf.clip_by_norm(cable_j.W, self.proj_weight_mag, axes=[self.param_axis]))
                #print("W.after :\n",tf.norm(cable_j.W,axis=self.param_axis))
                #cable_j.W.assign( cable_j.W / (tf.expand_dims(tf.norm(cable_j.W,axis=self.param_axis),axis=self.param_axis) + 1e-6) )
                # print(cable_j.name)
                # print("W:\n",cable_j.W)
                # print("b:\n",cable_j.b)
            for j in range(len(self.learnable_nodes)):
                node_j = self.learnable_nodes[j]
                node_j.Sigma.assign(tf.clip_by_norm(node_j.Sigma, self.proj_weight_mag, axes=[self.param_axis]))

    def calc_evolved_cable_updates(self, gamma_et=1.0, decay_et=0.0, rule_type="temp_diff", lambda_e=0.01):
        """
        Applies a slow evolution of specific target cables/synapses via a
        temporal difference rule that uses information from a set of source cables/synapses

        NOTE: this function is not tested/fully integrated yet

        Args:
            gamma_et:
            decay_et:
            rule_type: lra, temp_diff, hybrid (i.e., lra + weighted temp_diff)

        """
        delta = []
        for pair in self.evolved_cable_pairs:
            targ_name, src_name = pair
            targ_cable = self.cables.get(targ_name) # the cable to evolve
            src_cable = self.cables.get(src_name)

            # TODO: try temporal difference of z_t and z_t-1 for each layer and
            # compute change to weights (recurrent learning)

            # calc update to cable.W thru temporal difference rule: src.W_t - src.W_tm1
            # src_W_t = src_cable.W # src.W_t
            # src_W_tm1 = targ_cable.Wmem # src.W_tm1
            # if src_W_tm1 is None:
            #     src_W_tm1 = 0.0
            # dW = src_W_t - src_W_tm1 # temporal difference adjustment
            # dW = -tf.transpose(dW)

            if rule_type == "lra": # lra
                # LRA update rule
                e_n = src_cable.postact_node.extract(src_cable.postact_comp)
                zf = src_cable.preact_node.extract(src_cable.preact_comp)

                dW = -tf.matmul(zf, e_n, transpose_a=True)
                dW = tf.transpose(dW)
            elif rule_type == "hybrid":
                e_n = src_cable.postact_node.extract(src_cable.postact_comp)
                zf = src_cable.preact_node.extract(src_cable.preact_comp)
                dW = -tf.matmul(zf, e_n, transpose_a=True)
                dW = tf.transpose(dW)

                src_W_t = src_cable.W # src.W_t
                src_W_tm1 = targ_cable.Wmem # src.W_tm1
                if src_W_tm1 is None:
                    src_W_tm1 = 0.0
                dW2 = src_W_t - src_W_tm1 # temporal difference adjustment
                dW2 = tf.transpose(-dW2)

                targ_cable.Wmem = src_cable.W + 0

                dW = dW + dW2 * lambda_e
            else: # temp_diff
                # TD update rule
                # calc update thru temporal difference (TD) rule: src.W_t - src.W_tm1
                src_W_t = src_cable.W # src.W_t
                src_W_tm1 = targ_cable.Wmem # src.W_tm1
                if src_W_tm1 is None:
                    src_W_tm1 = 0.0
                dW = src_W_t - src_W_tm1 # temporal difference adjustment
                dW = tf.transpose(-dW)
                targ_cable.Wmem = src_cable.W + 0 # set memory of src.W_tm1 to be src.W_t

            dW = dW * gamma_et
            if decay_et > 0.0:
                dW = dW - targ_cable.W * decay_et
            delta.append( dW )

        return delta

    def clear(self):
        """
        Clears/deletes any persistent signals currently embedded w/in this graph's Nodes
        """
        for c in range(len(self.exec_cycles)):
            cycle_c = self.exec_cycles[c]
            for i in range(len(cycle_c)):
                cycle_c[i].clear()
