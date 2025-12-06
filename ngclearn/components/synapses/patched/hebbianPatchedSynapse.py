# %%

import matplotlib.pyplot as plt
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn

from ngcsimlib.logger import info
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

from ngclearn.components.synapses.patched import PatchedSynapse
from ngclearn.utils import tensorstats

# @partial(jit, static_argnums=[3, 4, 5, 6, 7, 8, 9])
def _calc_update(
        pre, post, W, mask, w_bound, is_nonnegative=True, signVal=1., prior_type=None, prior_lmbda=0., pre_wght=1.,
        post_wght=1.
):
    """
    Compute a tensor of adjustments to be applied to a synaptic value matrix.

    Args:
        pre: pre-synaptic statistic to drive Hebbian update

        post: post-synaptic statistic to drive Hebbian update

        W: synaptic weight values (at time t)

        mask: synaptic weight masking matrix (same shape as W)

        w_bound: maximum value to enforce over newly computed efficacies

        is_nonnegative: (Unused)

        signVal: multiplicative factor to modulate final update by (good for
            flipping the signs of a computed synaptic change matrix)

        prior_type: prior type or name (Default: None)

        prior_lmbda: prior parameter (Default: 0.0)

        pre_wght: pre-synaptic weighting term (Default: 1.)

        post_wght: post-synaptic weighting term (Default: 1.)

    Returns:
        an update/adjustment matrix, an update adjustment vector (for biases)
    """

    _pre = pre * pre_wght
    _post = post * post_wght
    dW = jnp.matmul(_pre.T, _post)
    db = jnp.sum(_post, axis=0, keepdims=True)
    dW_reg = 0.

    if w_bound > 0.:
        dW = dW * (w_bound - jnp.abs(W))

    if prior_type == "l2" or prior_type == "ridge":
        dW_reg = W

    if prior_type == "l1" or prior_type == "lasso":
        dW_reg = jnp.sign(W)

    if prior_type == "l1l2" or prior_type == "elastic_net":
        l1_ratio = prior_lmbda[1]
        prior_lmbda = prior_lmbda[0]
        dW_reg = jnp.sign(W) * l1_ratio + W * (1-l1_ratio)/2

    dW = dW + prior_lmbda * dW_reg

    if mask != None:
        dW = dW * mask

    return dW * signVal, db * signVal

# @partial(jit, static_argnums=[1,2, 3])
def _enforce_constraints(W, block_mask, w_bound, is_nonnegative=True):
    """
    Enforces constraints that the (synaptic) efficacies/values within matrix
    `W` must adhere to.

    Args:
        W: synaptic weight values (at time t)

         block_mask: weight mask matrix

        w_bound: maximum value to enforce over newly computed efficacies

        is_nonnegative: ensure updated value matrix is strictly non-negative

    Returns:
        the newly evolved synaptic weight value matrix
    """
    _W = W
    if w_bound > 0.:
        if is_nonnegative:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)

    if block_mask != None:
        _W = _W * block_mask

    return _W


class HebbianPatchedSynapse(PatchedSynapse):
    """
    A synaptic cable that adjusts its efficacies via a two-factor Hebbian
    adjustment rule.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal to drive first term of Hebbian update (takes in external signals)
    | post - post-synaptic signal to drive 2nd term of Hebbian update (takes in external signals)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies
    | dBiases - current delta vector containing changes to be applied to bias values
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        n_sub_models: The number of submodels in each layer (Default: 1 similar functionality as DenseSynapse)

        stride_shape: Stride shape of overlapping synaptic weight value matrix
            (Default: (0, 0))

        eta: global learning rate

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        block_mask: weight mask matrix

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)


        prior: a kernel to drive prior of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            prior to use and 2nd element as a floating point number
            calling the prior parameter lambda (Default: (None, 0.))
            currently it supports "l1" or "lasso" or "l2" or "ridge" or "l1l2" or "elastic_net".
            usage guide:
            prior = ('l1', 0.01) or prior = ('lasso', lmbda)
            prior = ('l2', 0.01) or prior = ('ridge', lmbda)
            prior = ('l1l2', (0.01, 0.01)) or prior = ('elastic_net', (lmbda, l1_ratio))


        sign_value: multiplicative factor to apply to final synaptic update before
            it is applied to synapses; this is useful if gradient descent style
            optimization is required (as Hebbian rules typically yield
            adjustments for ascent)

        optim_type: optimization scheme to physically alter synaptic values
            once an update is computed (Default: "sgd"); supported schemes
            include "sgd" and "adam"

            :Note: technically, if "sgd" or "adam" is used but `signVal = 1`,
                then the ascent form of each rule is employed (signVal = -1) or
                a negative learning rate will mean a descent form of the
                `optim_scheme` is being employed

        pre_wght: pre-synaptic weighting factor (Default: 1.)

        post_wght: post-synaptic weighting factor (Default: 1.)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure

        batch_size: the size of each mini batch
    """

    def __init__(
            self, name, shape, n_sub_models=1, stride_shape=(0,0), eta=0., weight_init=None, bias_init=None,
            block_mask=None, w_bound=1., is_nonnegative=False, prior=(None, 0.), sign_value=1., optim_type="sgd",
            pre_wght=1., post_wght=1., p_conn=1., resist_scale=1., batch_size=1, **kwargs
    ):
        super().__init__(
            name, shape, n_sub_models, stride_shape, block_mask, weight_init, bias_init, resist_scale, p_conn,
            batch_size=batch_size, **kwargs
        )

        prior_type, prior_lmbda = prior
        self.prior_type = prior_type
        self.prior_lmbda = prior_lmbda

        self.n_sub_models = n_sub_models
        self.sub_stride = stride_shape

        self.shape = (shape[0] + (2 * stride_shape[0]),
                      shape[1] + (2 * stride_shape[1]))
        self.sub_shape = (shape[0]//n_sub_models + (2 * stride_shape[0]),
                          shape[1]//n_sub_models + (2*  stride_shape[1]))

        ## synaptic plasticity properties and characteristics
        self.Rscale = resist_scale
        self.w_bound = w_bound
        self.pre_wght = pre_wght
        self.post_wght = post_wght
        self.eta = eta
        self.is_nonnegative = is_nonnegative
        self.sign_value = sign_value

        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.preVals = jnp.zeros((self.batch_size, self.shape[0]))
        self.postVals = jnp.zeros((self.batch_size, self.shape[1]))
        self.pre = Compartment(self.preVals)
        self.post = Compartment(self.postVals)
        self.block_mask = block_mask
        self.dWeights = Compartment(jnp.zeros(self.shape))
        self.dBiases = Compartment(jnp.zeros(self.shape[1]))

        #key, subkey = random.split(self.key.get())
        self.opt_params = Compartment(get_opt_init_fn(optim_type)(
            [self.weights.get(), self.biases.get()]
            if bias_init else [self.weights.get()]))

    @staticmethod
    def _compute_update(block_mask, w_bound, is_nonnegative, sign_value, prior_type, prior_lmbda, pre_wght,
                        post_wght, pre, post, weights):
        ## calculate synaptic update values
        dW, db = _calc_update(
            pre, post, weights, block_mask, w_bound, is_nonnegative=is_nonnegative,
            signVal=sign_value, prior_type=prior_type, prior_lmbda=prior_lmbda, pre_wght=pre_wght,
            post_wght=post_wght)

        return dW  * jnp.where(0 != jnp.abs(weights), 1, 0) , db

    @compilable
    def evolve(self):
        # Get the variables
        pre = self.pre.get()
        post = self.post.get()
        weights = self.weights.get()
        biases = self.biases.get()
        opt_params = self.opt_params.get()

        ## calculate synaptic update values
        dWeights, dBiases = HebbianPatchedSynapse._compute_update(
            self.block_mask, self.w_bound, self.is_nonnegative, self.sign_value, self.prior_type, self.prior_lmbda,
            self.pre_wght, self.post_wght, pre, post, weights
        )
        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if self.bias_init != None:
            opt_params, [weights, biases] = self.opt(opt_params, [weights, biases], [dWeights, dBiases])
        else:
            # ignore db since no biases configured
            opt_params, [weights] = self.opt(opt_params, [weights], [dWeights])
        ## ensure synaptic efficacies adhere to constraints
        weights = _enforce_constraints(weights, self.block_mask, self.w_bound, is_nonnegative=self.is_nonnegative)

        # Update compartments
        self.opt_params.set(opt_params)
        self.weights.set(weights)
        self.biases.set(biases)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(preVals) # inputs
        self.outputs.set(postVals) # outputs
        self.pre.set(preVals) # pre
        self.post.set(postVals) # post
        self.dWeights.set(jnp.zeros(self.shape)) # dW
        self.dBiases.set(jnp.zeros(self.shape[1])) # db


    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "HebbianPatchedSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted via two-term/factor Hebbian adjustment"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "pre": "Pre-synaptic statistic for Hebb rule (z_j)",
                 "post": "Post-synaptic statistic for Hebb rule (z_i)"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic weight value adjustment matrix produced at time t",
                 "dBiases": "Synaptic bias/base-rate value adjustment vector produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Overall shape of synaptic weight value matrix; number inputs x number outputs",
            "n_sub_models": "The number of submodels in each layer",
            "stride_shape": "Stride shape of overlapping synaptic weight value matrix",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "is_nonnegative": "Should synapses be constrained to be non-negative post-updates?",
            "sign_value": "Scalar `flipping` constant -- changes direction to Hebbian descent if < 0",
            "eta": "Global (fixed) learning rate",
            "pre_wght": "Pre-synaptic weighting coefficient (q_pre)",
            "post_wght": "Post-synaptic weighting coefficient (q_post)",
            "w_bound": "Soft synaptic bound applied to synapses post-update",
            "prior": "prior name and value for synaptic updating prior",
            "block_mask": "weight mask matrix",
            "optim_type": "Choice of optimizer to adjust synaptic weights"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] + b ;"
                            "dW_{ij}/dt = eta * [(z_j * q_pre) * (z_i * q_post)] - g(W_{ij}) * prior_lmbda",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = HebbianPatchedSynapse("Wab", (9, 30), 3, (0, 0), optim_type='adam',
                             sign_value=-1.0, prior=("l1l2", 0.001))
    print(Wab)
    plt.imshow(Wab.weights.get(), cmap='gray')
    plt.show()

