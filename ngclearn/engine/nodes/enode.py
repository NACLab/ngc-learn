import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils

class ENode(Node):
    """
    | Implements a (rate-coded) error node simplified to its fixed-point form:
    |   e = target - mu // in the case of squared error (Gaussian error units)
    |   e = signum(target - mu) // in the case of absolute error (Laplace error units)
    | where:
    |   target - a desired target activity value (target = pred_targ)
    |   mu - an external prediction signal of the target activity value (mu = pred_mu)

    | Compartments:
    |   * pred_mu - prediction signals (deposited signals summed)
    |   * pred_targ - target signals (deposited signals summed)
    |   * z - the error neural activities, set as z = e
    |   * phi(z) -  the post-activation of the error activities in z
    |   * L - the local loss represented by the error activities
    |   * avg_scalar - multiplies L and z by (1/avg_scalar)

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        error_type: type of distance/error measured by this error node. Setting this
            to "mse" will set up squared-error neuronal units (derived from
            L = 0.5 * ( Sum_j (target - mu)^2_j )), and "mae" will set up
            mean absolute error neuronal units (derived from L = Sum_j \|target - mu\| ).

        act_fx: activation function -- phi(v) -- to apply to error activities (Default = "identity")

        batch_size: batch-size this node should assume (for use with static graph optimization)

        precis_kernel: 2-Tuple defining the initialization of the precision weighting synapses that will
            modulate the error neural activities. For example, an argument could be: ("uniform", 0.01)
            The value types inside each slot of the tuple are specified below:

            :init_scheme (Tuple[0]): initialization scheme, e.g., "uniform", "gaussian".

            :init_scale (Tuple[1]): scalar factor controlling the scale/magnitude of initialization distribution, e.g., 0.01.

            :Note: specifying None will result in precision weighting being applied to the error neurons.
                Understand that care should be taken w/ respect to this particular argument as precision
                synapses involve an approximate inversion throughout simulation steps

        constraint_kernel: Dict defining the constraint type to be applied to the learnable parameters
            of this node. The expected keys and corresponding value types are specified below:

            :`'clip_type'`: type of clipping constraint to be applied to learnable parameters/synapses.
                If "norm_clip" is specified, then norm-clipping will be applied (with a check if the
                norm exceeds "clip_mag"), and if "forced_norm_clip" then norm-clipping will be applied
                regardless each time apply_constraint() is called.

            :`'clip_mag'`: the magnitude of the worse-case bounds of the clip to apply/enforce.

            :`'clip_axis'`: the axis along which the clipping is to be applied (to each matrix).

            :Note: specifying None will mean no constraints are applied to this node's parameters

        ex_scale: a scale factor to amplify error neuron signals (Default = 1)
    """
    def __init__(self, name, dim, error_type="mse", act_fx="identity", batch_size=1,
                 precis_kernel=None, constraint_kernel=None, ex_scale=1.0):
        node_type = "error"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.error_type = error_type
        self.ex_scale = ex_scale
        self.act_fx = act_fx
        self.fx = tf.identity
        self.dfx = None
        self.is_clamped = False

        self.compartment_names = ["pred_mu", "pred_targ", "z", "phi(z)", "L", "e"]# "weights", "avg_scalar"]
        self.compartments = {}
        for name in self.compartment_names:
            name_v = None
            if "phi(z)" in name:
                name_v = "{}_phi_z".format(self.name)
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name=name_v)
            elif "L" in name:
                name_v = "{}_{}".format(self.name, name)
                self.compartments[name] = tf.Variable(tf.zeros([1,1]), name=name_v)
            else:
                name_v = "{}_{}".format(self.name, name)
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name=name_v)
        self.mask_names = ["mask"]
        self.masks = {}
        for name in self.mask_names:
            self.masks[name] = tf.Variable(tf.ones([batch_size,dim]), name="{}_{}".format(self.name, name))

        self.connected_cables = []

        self.use_mod_factor = False

        # fixed-point error neurons can have associated with them a special precision
        # synaptic parameter matrix, as set up below (if a precision kernel is given as an argument)
        self.Prec = None # the precision matrix for this node
        self.Sigma = None # the covariance matrix for this node
        self.precis_kernel = precis_kernel
        if precis_kernel is not None: # (init_type, sigma)
            self.is_learnable = True # if precision is used, then this node becomes "learnable"
            prec_init_type, prec_sigma = precis_kernel
            # NOTE: we ignore "prec_init_type" for now b/c we want to ensure a uniform initialization is used
            #       for empirical stability reasons
            # create potential precision synapses at point of connection to target_node
            diag = tf.eye(self.dim) #* prec_sigma
            init = diag + tf.random.uniform([self.dim,self.dim],minval=-prec_sigma,maxval=prec_sigma) * (1.0 - diag)
            #init = diag + init_weights("orthogonal", [self.z_dims[l],self.z_dims[l]], stddev=prec_sigma, seed=seed) * (1.0 - diag)
            Sigma = tf.Variable(init, name="Sigma_{0}".format(self.name) )
            self.Sigma = Sigma
            #self.Prec = tf.Variable(tf.zeros([self.dim,self.dim]), name="Prec_{0}".format(self.name) )
        self.constraint_kernel = constraint_kernel

    def compile(self):
        info = super().compile()
        # we have to special re-compile the L compartment to be (1 x 1)
        self.compartments["L"] = tf.Variable(tf.zeros([1,1]), name="{}_L".format(self.name))

        info["error_type"] = self.error_type
        info["ex_scale"] = self.ex_scale
        info["phi(x)"] = self.act_fx
        if self.precis_kernel is not None:
            info["precision.form"] = self.precis_kernel
        return info

    def set_constraint(self, constraint_kernel):
        self.constraint_kernel = constraint_kernel

    def step(self, injection_table=None, skip_core_calc=False):
        bmask = self.masks.get("mask")
        Ws = self.compartments.get("weights")
        Ns = self.compartments.get("avg_scalar")

        ########################################################################
        if skip_core_calc == False:
            if self.is_clamped == False:
                # clear any relevant compartments that are NOT stateful before accruing
                # new deposits (this is crucial to ensure any desired stateless properties)
                if self.do_inplace == True:
                    self.compartments["pred_mu"].assign(self.compartments["pred_mu"] * 0)
                    self.compartments["pred_targ"].assign(self.compartments["pred_targ"] * 0)
                else:
                    self.compartments["pred_mu"] = (self.compartments["pred_mu"] * 0)
                    self.compartments["pred_targ"]= (self.compartments["pred_targ"] * 0)

                # gather deposits from any connected nodes & insert them into the
                # right compartments/regions -- deposits in this logic are linearly combined
                for cable in self.connected_cables:
                    deposit = cable.propagate()
                    dest_comp = cable.dest_comp
                    if self.do_inplace == True:
                        self.compartments[dest_comp].assign(self.compartments[dest_comp] + deposit)
                    else:
                        self.compartments[dest_comp] = (deposit + self.compartments[dest_comp])

                # core logic for the (node-internal) dendritic calculation
                # error neurons are a fixed-point result/calculation as below:
                pred_targ = self.compartments["pred_targ"]
                pred_mu = self.compartments["pred_mu"]

                # if self.name == "e0y":
                #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                #     print("pred_targ:\n",pred_targ.numpy())
                #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if bmask is not None:
                    pred_targ = pred_targ * bmask
                    pred_mu = pred_mu * bmask
                z = None
                L_batch = None
                L = None
                if self.error_type == "mse": # squared error neurons
                    z = e = pred_targ - pred_mu
                    # compute local loss that this error node represents
                    L_batch = tf.reduce_sum(e * e, axis=1, keepdims=True) #/(e.shape[0] * 2.0)
                elif self.error_type == "mae": # absolute error neurons
                    z = e = pred_targ - pred_mu
                    # compute local loss that this error node represents
                    L_batch = tf.reduce_sum(tf.math.abs(e), axis=1, keepdims=True) #/(e.shape[0] * 2.0)
                    z = e = tf.math.sign(e)
                else:
                    print("Error: {0} for error neuron not implemented yet".format(self.error_type))
                    sys.exit(1)
                # TODO: implement Spratling-style error neurons derived from KL divergence
                ## eps2 = 1e-2
                ## self.e = self.z/tf.math.maximum(eps2, self.z_mu)

                # finish error neuron and local loss calculation
                if Ws is not None: # optionally scale units by fixed external set of weights
                    L_batch = L_batch * Ws
                    z = z * Ws
                L = tf.reduce_sum(L_batch) # sum across dimensions
                z = z  * self.ex_scale # apply any fixed magnification
                if Ns is not None: # optionally scale units and local loss by 1/Ns
                    L = L * (1.0/Ns)
                    z = z * (1.0/Ns)
                if self.do_inplace == True:
                    self.compartments["e"].assign( z )
                else:
                    self.compartments["e"] = z

                if self.do_inplace == True:
                    self.compartments["L"].assign( [[L]] )
                else:
                    self.compartments["L"] = np.asarray([[L]])

                # apply error precision processing if configured
                if self.Prec is not None:
                    z = tf.matmul(z, self.Prec)
                if self.do_inplace == True:
                    self.compartments["z"].assign(z)
                else:
                    self.compartments["z"] = z

            # else, no deposits are accrued (b/c this node is hard-clamped to a signal)
            ########################################################################

        # apply post-activation non-linearity
        if self.do_inplace == True:
            self.compartments["phi(z)"].assign(self.fx(self.compartments["z"]))
        else:
            self.compartments["phi(z)"] = (self.fx(self.compartments["z"]))

        if bmask is not None: # applies mask to all component variables of this node
            for key in self.compartments:
                if "L" not in key:
                    if self.compartments.get(key) is not None:
                        if self.do_inplace == True:
                            self.compartments[key].assign( self.compartments.get(key) * bmask )
                        else:
                            self.compartments[key] = ( self.compartments.get(key) * bmask )

        ########################################################################
        self.t += 1

        # a node returns a list of its named component values
        values = []
        for comp_name in self.compartments:
            comp_value = self.compartments.get(comp_name)
            values.append((self.name, comp_name, comp_value))
        return values

    def compute_precision(self, rebuild_cov=True):
        """
            Co-function that pre-computes the precision matrices for this NGC node.
            NGC uses the Cholesky-decomposition form of precision (Sigma)^{-1}

            Args:
                rebuild_cov: rebuild the underlying covariance matrix after re-computing
                    precision (Default = True)
        """
        diag_l = tf.eye(self.Sigma.shape[1])
        # Note for Numerical Stability:
        #   Add small pertturbation eps * I to covariance before decomposing
        #   (due to rapidly decaying Eigen values)
        #R = tf.linalg.cholesky(cov_l + diag_l) # decompose
        if rebuild_cov is True:
            eps = 0.00025 #0.0001 # stability factor for precision/covariance computation
            cov_l = self.Sigma #tf.math.abs(self.Sigma[l])
            #diag_l = tf.eye(cov_l.shape[1])
            vari_l = tf.math.maximum(1.0, cov_l) * diag_l # restrict diag( Sigma ) to be >= 1.0
            cov_l = vari_l + (cov_l * (1.0 - diag_l))
            cov_l = cov_l + eps
            self.Sigma.assign( cov_l )
        else:
            cov_l = self.Sigma + 0
            self.Sigma.assign( cov_l )
        R = tf.linalg.cholesky(cov_l) # decompose
        prec_l = tf.transpose(tf.linalg.triangular_solve(R,diag_l,lower=True))
        self.Prec = prec_l
        #self.Prec.assign(prec_l)
        return R, prec_l

    def calc_update(self, update_radius=-1.0):
        delta = []
        # compute update to lateral correlation synapses

        # new rule for precision updates
        # if self.Sigma is not None:
        #     Prec_l = self.Prec
        #     e = self.compartments.get("z")
        #     weighted_e = self.compartments.get("z") # get weight prediction errors
        #     pr_val = 0.0002 #0.001
        #     dSigma = tf.eye(Prec_l.shape[0]) - tf.matmul(weighted_e, e, transpose_a=True)
        #     #Sigma = Sigma + dSigma * pr_val
        #     dW = -dSigma
        #     delta.append(dW)

        if self.Sigma is not None:
            Prec_l = self.Prec
            #e_noprec = self.stat.get("pre_z")
            #e = e_noprec
            e = self.compartments.get("phi(z)")
            B = tf.matmul(e, e, transpose_a=True)
            #dW = tf.matmul(tf.matmul(-Prec_l, B), Prec_l) - Prec_l
            dW = (B - Prec_l) * 0.5 # d_L_l / d_cov_l (derivative w.r.t. covariance)
            #dW = tf.matmul(tf.matmul(Prec_l, B), Prec_l) * 0.5 - Prec_l * 0.5 # deriv w.r.t. Cov, can get Prec from Cholesky/Triangular solve from Cov
            #dW = self.Sigma[l] - B * 0.5 # <-- direct derivative w.r.t. precision but would require Cov = (Prec)^{-1}
            if update_radius > 0.0:
                dW = tf.clip_by_norm(dW, update_radius)
            if self.use_mod_factor is True:
                W_M = transform_utils.calc_modulatory_factor(self.Sigma)
                dW = dW * W_M
            dW = -dW
            delta.append(dW)
        return delta

    def apply_constraints(self):
        """
        | Apply any constraints to the learnable parameters contained within
        | this cable. This function will execute any of the following
        | pre-configured constraints:
        | 1) compute new precision matrices
        | 2) project synapses to adhere to any embedded norm constraints
        """
        R, Prec_l = self.compute_precision()

        if self.constraint_kernel is not None:
            clip_type = self.constraint_kernel.get("clip_type")
            clip_mag = float(self.constraint_kernel.get("clip_mag"))
            clip_axis = int(self.constraint_kernel.get("clip_axis"))
            if clip_mag > 0.0: # apply constraints
                if clip_type == "norm_clip":
                    self.Sigma.assign(tf.clip_by_norm(self.Sigma, clip_mag, axes=[clip_axis]))
                elif clip_type == "forced_norm_clip":
                    _S = transform_utils.normalize_by_norm(self.Sigma, clip_mag, param_axis=clip_axis )
                    self.Sigma.assign(_S)
        return Prec_l
