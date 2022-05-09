import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils
from ngclearn.engine.cables.cable import Cable

class DCable(Cable):
    """
    A dense cable that transforms signals that travel across via a bundle of synapses.
    (In other words, a linear projection followed by an optional base-rate/bias shift.)

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

        w_kernel:

        b_kernel:

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
    def __init__(self, inp, out, w_kernel=None, b_kernel=None, shared_param_path=None,
                 clip_kernel=None, constraint_kernel=None, seed=69, name=None):
        cable_type = "dense"
        super().__init__(cable_type, inp, out, name, seed)

        self.gamma = 1.0
        self.use_mod_factor = False
        self.is_learnable = False
        self.shared_param_path = shared_param_path
        self.path_type = None
        self.w_kernel = w_kernel
        self.b_kernel = b_kernel
        self.coeff = 1.0
        self.clip_kernel = clip_kernel
        self.constraint_kernel = constraint_kernel

        self.W = None
        self.b = None
        if self.shared_param_path is not None:
            cable_to_mirror, path_type = shared_param_path
            """
                path_type = symm_tied, anti_symm_tied
            """
            self.path_type = path_type
            self.W = cable_to_mirror.W
            if self.path_type == "A": #"tied":
                self.b = cable_to_mirror.b
            else: # A^T or -A^T # symm_tied or anti_symm_tied
                if self.b_kernel is not None:
                    b = transform_utils.init_weights(kernel=self.b_kernel, shape=[1, out_dim], seed=self.seed)
                    #b = tf.zeros([1,out_dim])
                    self.b = tf.Variable(b, name="b_{0}".format(self.name))
        else:
            in_dim = self.src_node.dim
            out_dim = self.dest_node.dim
            init_type = w_kernel[0]
            if init_type == "lkwta":
                if in_dim != out_dim:
                    print("ERROR: input-side dim {0} != output-side dim {1}".format(in_dim,out_dim))
                    sys.exit(1)
                #lat_type = lateral_kernel.get("lat_type")
                n_group = w_kernel[1] #("n_group")
                alpha_scale = w_kernel[2] #("alpha_scale")
                beta_scale = w_kernel[3] #("beta_scale")
                #if n_group > 0:
                # create potential lateral competition synapses within this node
                W = transform_utils.create_competiion_matrix(in_dim, init_type, beta_scale, -alpha_scale, n_group, band=-1)
                self.W = tf.Variable( W, name="W_{0}".format(self.name) )
            else:
                W = transform_utils.init_weights(kernel=self.w_kernel, shape=[in_dim, out_dim], seed=self.seed)
                self.W = tf.Variable(W, name="W_{0}".format(self.name))
            if b_kernel is not None:
                b = transform_utils.init_weights(kernel=self.b_kernel, shape=[1, out_dim], seed=self.seed)
                #b = tf.zeros([1,out_dim])
                self.b = tf.Variable(b, name="b_{0}".format(self.name))
            #self.W_empty = self.W * 0

    def compile(self):
        """
        Executes the "compile()" routine for this cable.

        Returns:
            a dictionary containing post-compilation check information about this cable
        """
        info = super().compile()
        in_dim = self.src_node.dim
        out_dim = self.dest_node.dim
        W = self.W
        b = self.b
        if self.path_type is not None:
            info["path_type"] = self.path_type
            if self.path_type == "A^T" or self.path_type == "-A^T":
                W = tf.transpose(W)
            elif self.path_type == "A*A^T":
                W = tf.matmul(W,W,transpose_b=True)
        info["W.shape"] = W.shape
        if W.shape[0] != in_dim:
            print("ERROR: DCable {} self.W row_dim {} != out_dim {}".format(self.name, W.shape[0], in_dim))
        if W.shape[1] != out_dim:
            print("ERROR: DCable {} self.W col_dim {} != out_dim {}".format(self.name, W.shape[1], out_dim))
        if self.b is not None:
            info["b.shape"] = b.shape
            if b.shape[1] != out_dim:
                print("ERROR: DCable {} self.b col_dim {} != out_dim {}".format(self.name, W.shape[1], out_dim))
        return info


    def set_update_rule(self, preact, postact, gamma=1.0, use_mod_factor=False):
        preact_node, preact_comp = preact
        self.preact_node = preact_node
        self.preact_comp = preact_comp

        postact_node, postact_comp = postact
        self.postact_node = postact_node
        self.postact_comp = postact_comp

        self.gamma = gamma
        self.use_mod_factor = use_mod_factor
        self.is_learnable = True

    def propagate(self):
        in_signal = self.src_node.extract(self.src_comp) # extract input signal
        out_signal = None
        if self.shared_param_path is not None:
            if self.path_type == "A^T": # "symm_tied":
                out_signal = tf.matmul(in_signal, self.W, transpose_b=True)
                if self.b is not None:
                    out_signal = out_signal + self.b
            elif self.path_type == "-A^T": # "anti_symm_tied":
                out_signal = tf.matmul(in_signal, -self.W, transpose_b=True)
                if self.b is not None:
                    out_signal = out_signal + self.b
            elif self.path_type == "A*A^T":
                V = tf.matmul(self.W,self.W,transpose_b=True)
                V = tf.eye(V.shape[1]) - V
                out_signal = tf.matmul(in_signal, V)
            elif self.path_type == "A": # "tied":
                out_signal = tf.matmul(in_signal, self.W)
                if self.b is not None:
                    out_signal = out_signal + self.b
        else:
            out_signal = tf.matmul(in_signal, self.W)
            if self.b is not None:
                out_signal = out_signal + self.b
        out_signal = out_signal * self.coeff
        return out_signal # return output signal

    def calc_update(self):
        clip_type = "none"
        if self.clip_kernel is not None:
            self.clip_type = clip_kernel[0]
            self.clip_radius = clip_kernel[1]
        if self.is_learnable == True and self.shared_param_path is None:
            # Generalized Hebbian rule over arbitrary node signals
            postact_term = self.postact_node.extract(self.postact_comp)
            preact_term = self.preact_node.extract(self.preact_comp)
            dW = tf.matmul(preact_term, postact_term, transpose_a=True)
            if clip_type == "norm_clip":
                dW = tf.clip_by_norm(dW, clip_radius)
            elif clip_type == "hard_clip":
                dW = tf.clip_by_value(dW, -clip_radius, clip_radius)
            if self.use_mod_factor == True: # apply modulatory factor matrix to dW
                W_M = transform_utils.calc_modulatory_factor(self.W)
                dW = dW * W_M
            dW = -dW * self.gamma
            if self.b is not None:
                db = tf.reduce_sum(postact_term, axis=0, keepdims=True)
                if clip_type == "hard_clip":
                    db = tf.clip_by_value(db, -clip_radius, clip_radius)
                db = -db * self.gamma
                return [dW, db]
            return [dW]
        elif self.is_learnable == True:
            if self.b is not None:
                db = tf.reduce_sum(postact_term, axis=0, keepdims=True)
                if clip_type == "hard_clip":
                    db = tf.clip_by_value(db, -clip_radius, clip_radius)
                db = -db * self.gamma
                return [db]
        return []

    def apply_constraints(self):
        """
        | Apply any constraints to the learnable parameters contained within
        | this cable. This function will execute any of the following
        | pre-configured constraints:
        | 1) project weights to adhere to vector norm constraints
        """
        if self.constraint_kernel is not None:
            if self.is_learnable == True:
                clip_type = self.constraint_kernel.get("clip_type")
                clip_mag = float(self.constraint_kernel.get("clip_mag"))
                clip_axis = int(self.constraint_kernel.get("clip_axis"))
                if clip_mag > 0.0: # apply constraints
                    if clip_type == "norm_clip":
                        self.W.assign(tf.clip_by_norm(self.W, clip_mag, axes=[clip_axis]))
                    elif clip_type == "forced_norm_clip":
                        _W = transform_utils.normalize_by_norm(self.W, clip_mag, param_axis=clip_axis )
                        self.W.assign(_W)
