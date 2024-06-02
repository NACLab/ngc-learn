from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
import time

@partial(jit, static_argnums=[3, 4, 5, 6, 7, 8])
def calc_update(pre, post, W, w_bound, is_nonnegative=True, signVal=1., w_decay=0.,
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
def enforce_constraints(W, w_bound, is_nonnegative=True):
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

@jit
def compute_layer(inp, weight, biases, Rscale):
    """
    Applies the transformation/projection induced by the synaptic efficacie
    associated with this synaptic cable

    Args:
        inp: signal input to run through this synaptic cable

        weight: this cable's synaptic value matrix

        biases: this cable's bias value vector

        Rscale: scale factor to apply to synapses before transform applied
            to input values

    Returns:
        a projection/transformation of input "inp"
    """
    return jnp.matmul(inp, weight * Rscale) + biases

class HebbianSynapse(Component):
    """
    A synaptic cable that adjusts its efficacies via a two-factor Hebbian
    adjustment rule.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signal (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal to drive first term of Hebbian update
    | post - post-synaptic signal to drive 2nd term of Hebbian update
    | dW - current delta matrix containing changes to be applied to synaptic efficacies
    | db - current delta vector containing changes to be applied to bias value
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        bInit: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed Hebbian adjustment (Default: 0); note that decay is not
            applied to any configured biases

        signVal: multiplicative factor to apply to final synaptic update before
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

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, shape, eta=0., wInit=("uniform", 0., 0.3),
                 bInit=None, w_bound=1., is_nonnegative=False, w_decay=0.,
                 signVal=1., optim_type="sgd", pre_wght=1., post_wght=1.,
                 Rscale=1., key=None, directory=None, **kwargs):
        super().__init__(name, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.shape = shape
        self.Rscale = Rscale
        self.w_bounds = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.pre_wght = pre_wght
        self.post_wght = post_wght
        self.eta = eta
        self.wInit = wInit
        self.bInit = bInit
        self.is_nonnegative = is_nonnegative
        self.signVal = signVal

        self.batch_size = 1
        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        key = random.PRNGKey(time.time_ns()) if key is None else key

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.preVals = jnp.zeros((self.batch_size, shape[0]))
        self.postVals = jnp.zeros((self.batch_size, shape[1]))
        self.inputs = Compartment(self.preVals)
        self.outputs = Compartment(self.postVals)
        self.pre = Compartment(self.preVals)
        self.post = Compartment(self.postVals)
        self.dW = Compartment(jnp.zeros(shape))
        self.db = Compartment(jnp.zeros(shape[1]))

        key, subkey = random.split(key)
        self.weights = Compartment(initialize_params(subkey, wInit, shape))
        key, subkey = random.split(key)
        self.biases = Compartment(initialize_params(subkey, bInit, (1, shape[1])) if bInit else 0.0)
        self.opt_params = Compartment(get_opt_init_fn(optim_type)([self.weights.value, self.biases.value] if bInit else [self.weights.value]))

        # loading weights
        if directory is not None:
            self.load(directory)

        self.batch_size = 1

    @staticmethod
    def _advance_state(t, dt, Rscale, inputs, weights, biases):
        outputs = compute_layer(inputs, weights, biases, Rscale)
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _evolve(t, dt, opt, w_bounds, is_nonnegative, signVal, w_decay, pre_wght,
                post_wght, bInit, pre, post, weights, biases, dW, db, opt_params):
        dW, db = calc_update(pre, post,
                             weights, w_bounds, is_nonnegative=is_nonnegative,
                             signVal=signVal, w_decay=w_decay,
                             pre_wght=pre_wght, post_wght=post_wght)

        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if bInit != None:
            opt_params, [weights, biases] = opt(opt_params, [weights, biases], [dW, db])
        else:
            # ignore db since no biases configured
            opt_params, [weights] = opt(opt_params, [weights], [dW])
        ## ensure synaptic efficacies adhere to constraints
        weights = enforce_constraints(weights, w_bounds,
                                           is_nonnegative=is_nonnegative)
        return opt_params, weights, biases

    @resolver(_evolve)
    def evolve(self, opt_params, weights, biases):
        self.opt_params.set(opt_params)
        self.weights.set(weights)
        self.biases.set(biases)

    @staticmethod
    def _reset(batch_size, shape, wInit, bInit):
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
    def reset(self, inputs, outputs, pre, post, dW, db):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
        self.dW.set(dW)
        self.db.set(db)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.bInit != None:
            jnp.savez(file_name, weights=self.weights.value, biases=self.biases.value)
        else:
            jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set(data['weights'])
        if "biases" in data.keys():
            self.biases.set(data['biases'])

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
            signVal=-1.0, bInit=("constant", 0., 0.))
    print(Wab)
