import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils
from ngclearn.engine.cables.cable import Cable
from ngclearn.engine.cables.rules.hebb_rule import HebbRule

class DCable(Cable):
    """
    A dense cable that transforms signals that travel across via a bundle of synapses.
    (In other words, a linear projection followed by an optional base-rate/bias shift.)

    Note: a dense cable only contains two possible learnable parameters, "A" and "b"
    each with only two terms for their local Hebbian updates.

    Args:
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

        w_kernel: an N-Tuple defining type of scheme to randomly initialize weights.

            :scheme (Tuple[0]): triggers the type of initalization scheme, for example,
                "gaussian" will apply an elementwise Gaussian initialization.
                (See the documentation for init_weights() in ngclearn.utils.transform_utils
                for details on all the types of initializations and their string codes that can
                be used.)

            :scheme_arg1 (Tuple[1]): first argument to control the initialization (for many
                schemes, setting this value to 1.0 or even omitting it is acceptable given that
                this parameter is ignored, for example, in "unif_scale", the second argument
                would be ignored.)
                (See the documentation for init_weights() in ngclearn.utils.transform_utils
                for details on all the types of initializations and their extra arguments.)

            :scheme_arg2 (Tuple[2]): second argument to control the initialization -- this is
                generally only necessary to set in the case of lateral competition initialization
                schemes, such as in the case of "lkwta" which requires a 3-Tuple specified as
                follows: ("lkwta",alpha_scale,beta_scale) where alpha_scale controls the
                strength of self-excitation and beta_scale controls the strength
                of the cross-unit inhibition.

        b_kernel: 2-Tuple defining type of scheme to randomly initialize weights.

            :scheme (Tuple[0]): triggers the type of initalization scheme, for example,
                "gaussian" will apply an elementwise Gaussian initialization.
                (See the documentation for init_weights() in ngclearn.utils.transform_utils
                for details on all the types of initializations and their string codes that can
                be used.)

            :scheme_arg1 (Tuple[1]): first argument to control the initialization (for many
                schemes, setting this value to 1.0 or even omitting it is acceptable given that
                this parameter is ignored, for example, in "unif_scale", the second argument
                would be ignored.)
                (See the documentation for init_weights() in ngclearn.utils.transform_utils
                for details on all the types of initializations and their extra arguments.)

        shared_param_path:

        clip_kernel: 3-Tuple defining type of clipping to apply to calculated synaptic adjustments.

            :clip_type (Tuple[0]): type of clipping constraint to apply. If "hard_clip" is set, then
                a hard-clipping routine is applied (ignoring "clip_axis") while "norm_clip" clips
                by checking if the norm exceeds "clip_value" along "clip_axis". Note that "hard_clip"
                will also be applied to biases (while "clip_norm" is not).

            :clip_value (Tuple[1]): the magnitude of the worse-case bounds of the clip to apply/enforce.

            :clip_axis (Tuple[2]): the axis along which the clipping is to be applied (to each matrix).

            :Note: specifying None will mean no clipping is applied to this cable's calculated updates

        constraint_kernel: Dict defining the constraint type to be applied to the learnable parameters
            of this cable. The expected keys and corresponding value types are specified below:

            :`'clip_type'`: type of clipping constraint to be applied to learnable parameters/synapses.
                If "norm_clip" is specified, then norm-clipping will be applied (with a check if the
                norm exceeds "clip_mag"), and if "forced_norm_clip" then norm-clipping will be applied
                regardless each time apply_constraint() is called.

            :`'clip_mag'`: the magnitude of the worse-case bounds of the clip to apply/enforce.

            :`'clip_axis'`: the axis along which the clipping is to be applied (to each matrix).

            :Note: specifying None will mean no constraints are applied to this cable's parameters

        name: the string name of this cable (Default = None which creates an auto-name)

        seed: integer seed to control determinism of any underlying synapses
            associated with this cable

    """
    def __init__(self, inp, out, init_kernels=None, shared_param_path=None,
                 clip_kernel=None, constraint_kernel=None, seed=69, name=None):
        cable_type = "dense"
        super().__init__(cable_type, inp, out, name, seed)

        self.gamma = 1.0
        self.use_mod_factor = False
        self.is_learnable = False
        self.shared_param_path = shared_param_path
        self.path_type = None
        self.coeff = 1.0
        self.clip_kernel = clip_kernel
        self.constraint_kernel = constraint_kernel
        self.decay_kernel = None

        self.params["A"] = None
        self.update_terms["A"] = None
        self.params["b"] = None
        self.update_terms["b"] = None

        in_dim = self.src_node.dim
        out_dim = self.dest_node.dim

        if self.shared_param_path is not None:
            cable_to_mirror, path_type = shared_param_path
            """
                path_type = A, A^T, A+b, -A^T, A^T+b, -A^T+b
            """
            self.path_type = path_type
            A = cable_to_mirror.params["A"]
            b = cable_to_mirror.params["b"]
            if "A" in self.path_type: # share A matrix
                self.params["A"] = A
            if "b" in self.path_type: # share b bias vector
                self.params["b"] = b

        if init_kernels is not None:
            A_init = init_kernels.get("A_init") # get A's init schem
            if A_init is not None and self.params.get("A") is None:
                scheme = A_init[0] # N-tuple specifying init scheme
                if scheme == "lkwta":
                    if in_dim != out_dim:
                        print("ERROR: input-side dim {0} != output-side dim {1}".format(in_dim,out_dim))
                        sys.exit(1)
                    n_group = A_init[1] #("n_group")
                    alpha_scale = A_init[2] #("alpha_scale")
                    beta_scale = A_init[3] #("beta_scale")
                    # create potential lateral competition synapses within this node
                    A = transform_utils.create_competiion_matrix(in_dim, scheme, beta_scale, -alpha_scale, n_group, band=-1)
                    A = tf.Variable( A, name="A_{0}".format(self.name) )
                    self.params["A"] = A
                else:
                    A = transform_utils.init_weights(kernel=A_init, shape=[in_dim, out_dim], seed=self.seed)
                    A = tf.Variable(A, name="A_{0}".format(self.name))
                    self.params["A"] = A
            b_init = init_kernels.get("b_init") # get b's init scheme
            if b_init is not None and self.params.get("b") is None:
                #scheme = b_init[0] # N-tuple specifying init scheme
                b = transform_utils.init_weights(kernel=b_init, shape=[1, out_dim], seed=self.seed)
                b = tf.Variable(b, name="b_{0}".format(self.name))
                self.params["b"] = b
        if self.path_type is None:
            self.path_type = "none"

    def compile(self):
        """
        Executes the "compile()" routine for this cable.

        Returns:
            a dictionary containing post-compilation check information about this cable
        """
        info = super().compile()
        if self.path_type is None:
            self.path_type = "none"
        in_dim = self.src_node.dim
        out_dim = self.dest_node.dim
        A = self.params.get("A")
        b = self.params.get("b")
        if self.path_type is not None:
            info["path_type"] = self.path_type
            if A is not None and self.path_type is not None:
                if "A^T" in self.path_type:
                    A = tf.transpose(A)
                elif self.path_type == "A*A^T":
                    A = tf.matmul(A,A,transpose_b=True)
        if A is not None:
            info["A.shape"] = A.shape
            if A.shape[0] != in_dim:
                print("ERROR: DCable {} self.A row_dim {} != out_dim {}".format(self.name, A.shape[0], in_dim))
            if A.shape[1] != out_dim:
                print("ERROR: DCable {} self.A col_dim {} != out_dim {}".format(self.name, A.shape[1], out_dim))
        if b is not None:
            info["b.shape"] = b.shape
            if b.shape[1] != out_dim:
                print("ERROR: DCable {} self.b col_dim {} != out_dim {}".format(self.name, b.shape[1], out_dim))
        return info

    def get_params(self, only_learnable=False):
        cable_theta = []
        for pname in self.params:
            if only_learnable == True:
                if self.update_terms.get([pname]) is not None:
                    cable_theta.append(self.params[pname])
            else:
                cable_theta.append(self.params[pname])
        return cable_theta

    def set_update_rule(self, preact=None, postact=None, update_rule=None, gamma=1.0,
                        use_mod_factor=False, param=None, decay_kernel=None):
        if update_rule is not None: # set user-specified update rule
            if param is not None:
                for pname in param:
                    _rule = update_rule.clone()
                    _rule.point_to_cable(self, pname)
                    self.update_terms[pname] = _rule
        else: # create default Hebbian update rule
            if preact is None and postact is None:
                print("ERROR: Both preact and postact CANNOT be None for {}".format(self.name))
                sys.exit(1)
            if param is not None:
                for pname in param:
                    update_rule = HebbRule()
                    update_rule.set_terms(terms=[preact, postact])
                    update_rule.point_to_cable(self, pname)
                    self.update_terms[pname] = update_rule
            else:
                print("ERROR: *param* target cannot be None for {}".format(self.name))
                sys.exit(1)
        self.gamma = gamma
        self.use_mod_factor = use_mod_factor
        self.is_learnable = True
        if self.decay_kernel is None:
            self.decay_kernel = decay_kernel

    def propagate(self):
        in_signal = self.src_node.extract(self.src_comp) # extract input signal
        out_signal = None
        A = self.params.get("A")
        b = self.params.get("b")
        if A is None and b is None:
            tf.print("ERROR: Both *A* and *b* CANNOT be None for {}".format(self.name))
            sys.exit(1)

        if self.path_type is not None:
            if self.path_type == "A*A^T":
                # Case 1: lateral cross-inhibition shared pattern
                V = tf.matmul(A,A,transpose_b=True)
                V = tf.eye(V.shape[1]) - V
                out_signal = tf.matmul(in_signal, V)
                if b is not None: # (A * input) + b
                    out_signal = out_signal + b
                return out_signal # terminal - return output here
        #print("A ",A.shape)
        #sys.exit(0)
        if A is not None: # Case 2a: A * input
            if self.path_type == "A^T":
                out_signal = tf.matmul(in_signal, A, transpose_b=True)
            elif self.path_type == "-A^T":
                out_signal = tf.matmul(in_signal, -A, transpose_b=True)
            else: # self.path_type == "none" or "A" or "A+b"
                out_signal = tf.matmul(in_signal, A)
        if b is not None: # Apply bias if applicable
            if out_signal is None:
                out_signal = in_signal
            out_signal = out_signal + b

        out_signal = out_signal * self.coeff
        return out_signal # return output signal

    def calc_update(self):
        delta = []

        clip_type = "none"
        if self.clip_kernel is not None:
            self.clip_type = clip_kernel[0]
            self.clip_radius = clip_kernel[1]

        A = self.params.get("A")
        b = self.params.get("b")

        dA = None
        db = None
        if self.is_learnable == True:
            if A is not None:
                A_update_rule = self.update_terms.get("A")
                if A_update_rule is not None:
                    # preact_node, preact_comp = preact
                    # postact_node, postact_comp = postact
                    # postact_term = postact_node.extract(postact_comp)
                    # preact_term = preact_node.extract(preact_comp)
                    #
                    # dA = tf.matmul(preact_term, postact_term, transpose_a=True)
                    dA = A_update_rule.calc_update()
                    if clip_type == "norm_clip":
                        dA = tf.clip_by_norm(dA, clip_radius)
                    elif clip_type == "hard_clip":
                        dA = tf.clip_by_value(dA, -clip_radius, clip_radius)
                    if self.use_mod_factor == True: # apply modulatory factor matrix to dA
                        A_M = transform_utils.calc_modulatory_factor(A)
                        dA = dA * A_M
                    dA = -dA * self.gamma

            if b is not None:
                b_update_rule = self.update_terms.get("b")
                if b_update_rule is not None:
                    if b is not None:
                        #preact_node, preact_comp = preact
                        # postact_node, postact_comp = postact
                        # postact_term = postact_node.extract(postact_comp)
                        # db = tf.reduce_sum(postact_term, axis=0, keepdims=True)
                        db = b_update_rule.calc_update(for_bias=True)
                        if clip_type == "hard_clip":
                            db = tf.clip_by_value(db, -clip_radius, clip_radius)
                        db = -db * self.gamma
        if dA is not None:
            delta.append(dA)
        if db is not None:
            delta.append(db)
        return delta

    def apply_constraints(self):
        """
        | Apply any constraints to the learnable parameters contained within
        | this cable. This function will execute any of the following
        | pre-configured constraints:
        | 1) project weights to adhere to vector norm constraints
        | 2) apply weight decay (to non-bias synaptic matrices)
        """
        ## apply synaptic constraints
        if self.constraint_kernel is not None:
            if self.is_learnable == True:
                clip_type = self.constraint_kernel.get("clip_type")
                clip_mag = float(self.constraint_kernel.get("clip_mag"))
                clip_axis = int(self.constraint_kernel.get("clip_axis"))
                if clip_mag > 0.0: # apply constraints
                    A = self.params.get("A")
                    if A is not None:
                        if clip_type == "norm_clip":
                            A.assign(tf.clip_by_norm(A, clip_mag, axes=[clip_axis]))
                        elif clip_type == "forced_norm_clip":
                            _A = transform_utils.normalize_by_norm(A, clip_mag, param_axis=clip_axis )
                            A.assign(_A)
        ## apply synaptic weight decay
        if self.decay_kernel is not None:
            decay_type = self.decay_kernel[0]
            decay_coeff = self.decay_kernel[1]
            if decay_coeff > 0.0:
                A = self.params.get("A")
                if decay_type == "l2":
                    factor = A * 0.5 # derivative of 0.5 * A^2
                    A.assign_sub(factor * decay_coeff)
                elif decay_type == "l1":
                    factor = tf.math.sign(A) # derivative of |A|
                    A.assign_sub(factor * decay_coeff)
