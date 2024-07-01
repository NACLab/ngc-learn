from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

@partial(jit, static_argnums=[3, 4, 5, 6, 7, 8])
def _calc_update(pre, post, W, w_bound, is_nonnegative=True, signVal=1., w_decay=0.,
                 pre_wght=1., post_wght=1.):
    """
    Compute a tensor of adjustments to be applied to a synaptic value matrix.

    Args:
        pre: pre-synaptic statistic to drive Hebbian update

        post: post-synaptic statistic to drive Hebbian update

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        is_nonnegative: (Unused)

        signVal: multiplicative factor to modulate final update by (good for
            flipping the signs of a computed synaptic change matrix)

        w_decay: synaptic decay factor to apply to this update

        pre_wght: pre-synaptic weighting term (Default: 1.)

        post_wght: post-synaptic weighting term (Default: 1.)

    Returns:
        an update/adjustment matrix, an update adjustment vector (for biases)
    """
    _pre = pre * pre_wght
    _post = post * post_wght
    dW = jnp.matmul(_pre.T, _post)
    db = jnp.sum(_post, axis=0, keepdims=True)
    if w_bound > 0.:
        dW = dW * (w_bound - jnp.abs(W))
    if w_decay > 0.:
        dW = dW - W * w_decay
    return dW * signVal, db * signVal

@partial(jit, static_argnums=[1,2])
def _enforce_constraints(W, w_bound, is_nonnegative=True):
    """
    Enforces constraints that the (synaptic) efficacies/values within matrix
    `W` must adhere to.

    Args:
        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        is_nonnegative: ensure updated value matrix is strictly non-negative

    Returns:
        the newly evolved synaptic weight value matrix
    """
    _W = W
    if w_bound > 0.:
        if is_nonnegative == True:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W

class HebbianSynapse(DenseSynapse):
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
    | dWweights - current delta matrix containing changes to be applied to synaptic efficacies
    | dBiases - current delta vector containing changes to be applied to bias values
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed Hebbian adjustment (Default: 0); note that decay is not
            applied to any configured biases

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
    """

    # Define Functions
    def __init__(self, name, shape, eta=0., weight_init=None, bias_init=None,
                 w_bound=1., is_nonnegative=False, w_decay=0., sign_value=1.,
                 optim_type="sgd", pre_wght=1., post_wght=1., p_conn=1.,
                 resist_scale=1., batch_size=1, **kwargs):
        super().__init__(name, shape, weight_init, bias_init, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.shape = shape
        self.Rscale = resist_scale
        self.w_bound = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.pre_wght = pre_wght
        self.post_wght = post_wght
        self.eta = eta
        self.is_nonnegative = is_nonnegative
        self.sign_value = sign_value

        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.preVals = jnp.zeros((self.batch_size, shape[0]))
        self.postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre = Compartment(self.preVals)
        self.post = Compartment(self.postVals)
        self.dWeights = Compartment(jnp.zeros(shape))
        self.dBiases = Compartment(jnp.zeros(shape[1]))

        #key, subkey = random.split(self.key.value)
        self.opt_params = Compartment(get_opt_init_fn(optim_type)(
            [self.weights.value, self.biases.value]
            if bias_init else [self.weights.value]))

    @staticmethod
    def _compute_update(w_bound, is_nonnegative, sign_value, w_decay, pre_wght,
                        post_wght, pre, post, weights):
        ## calculate synaptic update values
        dW, db = _calc_update(
            pre, post, weights, w_bound, is_nonnegative=is_nonnegative,
            signVal=sign_value, w_decay=w_decay, pre_wght=pre_wght,
            post_wght=post_wght)
        return dW, db

    @staticmethod
    def _evolve(opt, w_bound, is_nonnegative, sign_value, w_decay, pre_wght,
                post_wght, bias_init, pre, post, weights, biases, opt_params):
        ## calculate synaptic update values
        dWeights, dBiases = HebbianSynapse._compute_update(
            w_bound, is_nonnegative, sign_value, w_decay, pre_wght, post_wght,
            pre, post, weights
        )
        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if bias_init != None:
            opt_params, [weights, biases] = opt(opt_params, [weights, biases], [dWeights, dBiases])
        else:
            # ignore db since no biases configured
            opt_params, [weights] = opt(opt_params, [weights], [dWeights])
        ## ensure synaptic efficacies adhere to constraints
        weights = _enforce_constraints(weights, w_bound, is_nonnegative=is_nonnegative)
        return opt_params, weights, biases, dWeights, dBiases

    @resolver(_evolve)
    def evolve(self, opt_params, weights, biases, dWeights, dBiases):
        self.opt_params.set(opt_params)
        self.weights.set(weights)
        self.biases.set(biases)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            jnp.zeros(shape), # dW
            jnp.zeros(shape[1]), # db
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post, dWeights, dBiases):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "HebbianSynapse - performs an adaptable synaptic "
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
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
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
            "w_decay": "Synaptic decay term",
            "optim_type": "Choice of optimizer to adjust synaptic weights"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] + b ;"
                            "dW_{ij}/dt = eta * [(z_j * q_pre) * (z_i * q_post)] - W_{ij} * w_decay",
                "hyperparameters": hyperparams}
        return info

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = HebbianSynapse("Wab", (2, 3), 0.0004, optim_type='adam',
                             sign_value=-1.0, bias_init=("constant", 0., 0.))
    print(Wab)
