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

        name: str name of this node

        dim: number of neurons this node will contain
    """
    def __init__(self, node_type, name, dim):
        self.node_type = node_type
        self.name = name
        self.dim = dim
        self.batch_size = 1
        self.t = 0.0 # tracks this node's current notion of time

        self.is_learnable = False
        self.is_clamped = False
        self.compartment_names = None
        self.compartments = None
        self.constant_names = None
        self.constants = None
        self.mask_names = None
        self.masks = None
        self.connected_cables = []
        self.do_inplace = True # forces variables to be overriden/set in-place in memory

    def set_status(self, status=("static",1)):
        """
        Sets the status of this node to be either "static" or "dynamic".

        Note: Making this node "dynamic" in the sense that it can handle mini-batches of
        samples of arbitrary length BUT you CANNOT use "use_graph_optim = True"
        static-graph acceleration used w/in the NGCGraph .settle() routine, meaning
        your simulation run slower than when using acceleration.

        Args:
            status: 2-tuple where 1st element contains a string status flag and the
                2nd element contains the (integer) batch_size. If status is set
                to "dynamic", the second argument is arbitrary (setting it to 1 is
                sufficient), and if status is set to "static" you MUST choose
                a fixed batch_size that you will use w.r.t. this node.
        """
        if status[0] == "dynamic":
            self.do_inplace = False
            self.batch_size = status[1]
        else:
            self.do_inplace = True
            self.batch_size = status[1]

    def compile(self):
        """
        Executes the "compile()" routine for this node. Sub-class nodes can
        extend this in case they contain other elements besides compartments
        that must be configured properly for global simulation usage.

        Returns:
            a dictionary containing post-compilation check information about this cable
        """
        info = {}
        # set all variables & masks for each compartment to ensure they adhere to .batch_size
        for cname in self.compartment_names:
            curr_comp = self.compartments.get(cname)
            if curr_comp is not None:
                comp_dim = curr_comp.shape[1]
                comp_var_name = curr_comp.name
                comp_var_name = comp_var_name[0:curr_comp.name.index(":"):1]
                if self.do_inplace == True:
                    self.compartments[cname] = \
                        tf.Variable(tf.zeros([self.batch_size,comp_dim]), name=comp_var_name)
                else:
                    self.compartments[cname] = tf.zeros([self.batch_size,comp_dim])
        for mname in self.mask_names:
            curr_mask = self.masks.get(mname)
            if curr_mask is not None:
                mask_var_name = curr_mask.name
                mask_var_name = mask_var_name[0:curr_mask.name.index(":"):1]
                mask_dim = curr_mask.shape[1]
                if self.do_inplace == True:
                    self.masks[mname] = tf.Variable(tf.ones([self.batch_size,mask_dim]), name=mask_var_name)
                else:
                    self.masks[mname] = tf.ones([self.batch_size,mask_dim])
        parents = []
        for parent_cable in self.connected_cables:
            parents.append((parent_cable.src_node.name, parent_cable.src_comp))
        info["parents"] = parents
        info["object_type"] = self.node_type
        info["object_name"] = self.name
        info["n_connected_cables"] = len(self.connected_cables)
        info["n_compartments"] = len(self.compartments)
        info["compartments"] = self.compartment_names
        info["n_masks"] = len(self.masks)
        info["masks"] = self.mask_names
        if self.constants is not None:
            info["n_constants"] = len(self.constants)
            info["constants"] = self.constant_names
        info["do_inplace"] = self.do_inplace
        info["batch_size"] = self.batch_size

        return info

    def set_constraint(self, constraint_kernel):
        pass

    def wire_to(self, dest_node, src_comp, dest_comp, cable_kernel=None,
                mirror_path_kernel=None, name=None, short_name=None):
        """
        A wiring function that connects this node to another external node via a cable (or synaptic bundle)

        Args:
            dest_node: destination node (a Node object) to wire this node to

            src_comp: name of the compartment inside this node to transmit a signal from (to destination node)

            dest_comp: name of the compartment inside the destination node to transmit a signal to

            cable_kernel: Dict defining how to initialize the cable that will connect this node to the destination node.
                The expected keys and corresponding value types are specified below:

                :`'type'`: type of cable to be created.
                    If "dense" is specified, a DCable (dense cable/bundle/matrix of synapses) will be used to
                    transmit/transform information along.

                :`'init_kernels'`: a Dict specifying how parameters w/in the learnable parts of the cable are to
                    randomly initialized

                :`'seed'`: integer seed to deterministically control initialization of synapses in a DCable

                :Note: either cable_kernel, mirror_path_kernel MUST be set to something that is not None

            mirror_path_kernel: 2-Tuple that allows a currently existing cable to be re-used as a transformation.
                The value types inside each slot of the tuple are specified below:

                :cable_to_reuse (Tuple[0]): target cable (usually an existing DCable object) to shallow copy and mirror

                :mirror_type (Tuple[1]): how should the cable be mirrored? If "symm_tied" is specified, then the transpose
                    of this cable will be used to transmit information from this node to a destination node, if "anti_symm_tied"
                    is specified, the negative transpose of this cable will be used, and if "tied" is specified,
                    then this cable will be used exactly in the same way it was used in its source cable.

                :Note: either cable_kernel, mirror_path_kernel MUST be set to something that is not None

            name: the string name to be assigned to the generated cable (Default = None)

                :Note: setting this to None will trigger the created cable to auto-name itself
        """
        if cable_kernel is None and mirror_path_kernel is None:
            print("Error: Must either set |cable_kernel| or |mirror_path_kernel| argument! for node({})".format(self.name))
            sys.exit(1)
        init_kernels = None
        if cable_kernel is not None:
            init_kernels = cable_kernel.get("init_kernels")
        if mirror_path_kernel is not None: # directly share/shallow copy this cable
            #bias_init = cable_kernel.get("bias_init") # <--- for RBMs
            cable = DCable(inp=(self,src_comp),out=(dest_node,dest_comp), shared_param_path=mirror_path_kernel,
                           init_kernels=init_kernels, name=name)
        else:
            cable_type = cable_kernel.get("type")
            coeff = cable_kernel.get("coeff")
            if cable_type == "dense":
                seed = cable_kernel.get("seed")
                if seed is None:
                    seed = 69
                cable = DCable(inp=(self, src_comp), out=(dest_node, dest_comp), init_kernels=init_kernels,
                               seed=seed, name=name)
                if coeff is not None:
                    cable.coeff = coeff
            else:
                cable = SCable(inp=(self,src_comp),out=(dest_node,dest_comp), coeff=coeff, name=name)
        dest_node.connected_cables.append(cable)
        if short_name is not None:
            cable.short_name = short_name
        return cable

    def extract(self, comp_name):
        """
        Extracts the data signal value that is currently stored inside of a target compartment

        Args:
            comp_name: the name of the compartment in this node to extract data from
        """
        return self.compartments[comp_name]

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
        comp_name, signal = data
        # if is_persistent is True:
        #     ptr = self.comp_map[comp_name]
        #     self.injected[ptr] = 2 # 2 means purely clamped
        # else:
        #     ptr = self.comp_map[comp_name]
        #     self.injected[ptr] = 1  # 1 means this value can change w/ dynamics

        if self.do_inplace == True:
            self.compartments[comp_name].assign(signal)
        else:
            self.compartments[comp_name] = (signal)
        if is_persistent is True:
            self.is_clamped = True

    def inject(self, data):
        """
        Injects an externally provided named value (a vector/matrix) to the desired
        compartment within this node.

        Args:
            data: 2-Tuple containing a named external signal to clamp

                :compartment_name (Tuple[0]): the (str) name of the compartment to clamp this data signal to.

                :signal (Tuple[1]): the data signal block to clamp to the desired compartment name
        """
        self.clamp(data, is_persistent=False)

    def step(self, injection_table=None, skip_core_calc=False):
        """
        Executes this nodes internal integration/calculation for one discrete step
        in time, i.e., runs simulation of this node for one time step.

        Args:
            injection_table:

            skip_core_calc: skips the core components of this node's calculation (Default = False)
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

    def clear(self, batch_size=-1):
        """ Wipes/clears values of each compartment in this node (and sets .is_clamped = False). """
        #print("CLEAR for {} w/ ip = {}".format(self.name, self.do_inplace))
        #tf.print("=============== CLEAR ===============")
        self.t = 0.0
        self.is_clamped = False

        for comp_name in self.compartment_names:
            comp_value = self.compartments.get(comp_name)
            if comp_value is not None:
                if self.do_inplace == True:
                    self.compartments[comp_name].assign(comp_value * 0)
                else:
                    if batch_size > 0:
                        self.compartments[comp_name] = tf.zeros([batch_size, comp_value.shape[1]])
                    else:
                        self.compartments[comp_name] = (comp_value * 0)
        for mask_name in self.mask_names:
            mask_value = self.masks.get(mask_name)
            if mask_value is not None:
                if self.do_inplace == True:
                    self.masks[mask_name].assign(mask_value * 0 + 1)
                else:
                    if batch_size > 0:
                        self.masks[mask_name] = tf.ones([batch_size, mask_value.shape[1]])
                    else:
                        self.masks[mask_name] = (mask_value * 0 + 1)

    def set_cold_state(self, injection_table=None, batch_size=-1):
        """
        Sets each compartment to its cold zero-state of shape (batch_size x D).
        Note that this fills each vector/matrix state of each compartment to
        all zero values.

        Args:
            injection_table:

            batch_size: the axis=0 dimension of each compartment @ its cold zero-state
        """
        if injection_table is None:
            injection_table = {}
        batch_size_ = batch_size
        if batch_size_ <= 0:
            batch_size_ = self.batch_size
        for comp_name in self.compartment_names:
            #if self.injected.get(comp_name) is None:
            if injection_table.get(comp_name) is None:
                comp_value = self.compartments.get(comp_name)
                if comp_value is not None:
                    if comp_value.shape[0] > 1:
                        zero_state = tf.zeros([batch_size_, comp_value.shape[1]])
                    else:
                        zero_state = tf.zeros([1, comp_value.shape[1]])
                    if self.do_inplace == True:
                        self.compartments[comp_name].assign(zero_state + 0)
                    else:
                        self.compartments[comp_name] = (zero_state + 0)
        for comp_name in self.mask_names:
            #ptr = self.comp_map[comp_name]
            #if self.injected.get(comp_name) is None:
            #if self.injected[ptr] == 0:
            comp_value = self.masks.get(comp_name)
            if comp_value is not None:
                if comp_value.shape[0] > 1:
                    zero_state = tf.ones([batch_size_, comp_value.shape[1]])
                else:
                    zero_state = tf.ones([1, comp_value.shape[1]])
                if self.do_inplace == True:
                    self.masks[comp_name].assign(zero_state + 0)
                else:
                    self.masks[comp_name] = (zero_state + 0)

    def extract_params(self):
        return []

    def deep_store_state(self):
        """
        Performs a deep copy of all compartment statistics.

        Returns:
            Dict containing a deep copy of each named compartment of this node
        """
        stat_cpy = {}
        for key in self.compartments:
            value = self.compartments.get(key)
            if value is not None:
                stat_cpy[key] = value + 0
        return stat_cpy
