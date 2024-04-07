from ngcsimlib.component import Component
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params
from ngclearn.utils.optim import SGD, Adam
import time

@partial(jit, static_argnums=[3,4,5,6])
def calc_update(pre, post, W, w_bound, is_nonnegative=True, signVal=1., w_decay=0.):
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

    Returns:
        an update/adjustment matrix, an update adjustment vector (for biases)
    """
    dW = jnp.matmul(pre.T, post)
    db = jnp.sum(post + 0, axis=0, keepdims=True)
    if w_bound > 0.:
        dW = dW * (w_bound - jnp.abs(W))
    if w_decay > 0.:
        dW = dW - W * w_decay # jnp.matmul((1. - pre).T, (1. - post)) * w_decay
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
def apply_decay(dW, pre_s, post_s, w_decay):
    _dW = dW - jnp.matmul((1. - pre_s).T, (1. - post_s)) * w_decay
    return _dW

@jit
def compute_layer(inp, weight, biases):
    """
    Applies the transformation/projection induced by the synaptic efficacie
    associated with this synaptic cable

    Args:
        inp: signal input to run through this synaptic cable

        weight: this cable's synaptic value matrix

        biases: this cable's bias value vector

    Returns:
        a projection/transformation of input "inp"
    """
    return jnp.matmul(inp, weight) + biases

class HebbianSynapse(Component):
    """
    A synaptic cable that adjusts its efficacies via a two-factor Hebbian
    adjustment rule.

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

        signVal: multiplicative factor to apply to final synaptic update before
            it is applied to synapses; this is useful if gradient descent schemes
            are to be applied (as Hebbian rules typically yield adjustments for
            ascent)

        optim_type: optimization scheme to physically alter synaptic values
            once an update is computed (Default: "sgd"); supported schemes
            include "sgd" and "adam"

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'in'

    @classmethod
    def outputCompartmentName(cls):
        return 'out'

    @classmethod
    def triggerName(cls):
        return 'trigger'

    @classmethod
    def presynapticCompartmentName(cls):
        return 'pre'

    @classmethod
    def postsynSpikeName(cls):
        return 's_post'

    @classmethod
    def presynSpikeName(cls):
        return 's_pre'

    @classmethod
    def postsynapticCompartmentName(cls):
        return 'post'

    ## Bind Properties to Compartments for ease of use
    @property
    def trigger(self):
        return self.compartments.get(self.triggerName(), None)

    @trigger.setter
    def trigger(self, x):
        self.compartments[self.triggerName()] = x

    @property
    def inputCompartment(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, x):
        self.compartments[self.inputCompartmentName()] = x

    @property
    def outputCompartment(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @outputCompartment.setter
    def outputCompartment(self, x):
        self.compartments[self.outputCompartmentName()] = x

    @property
    def presynapticCompartment(self):
        return self.compartments.get(self.presynapticCompartmentName(), None)

    @presynapticCompartment.setter
    def presynapticCompartment(self, x):
        self.compartments[self.presynapticCompartmentName()] = x

    @property
    def postsynapticCompartment(self):
        return self.compartments.get(self.postsynapticCompartmentName(), None)

    @postsynapticCompartment.setter
    def postsynapticCompartment(self, x):
        self.compartments[self.postsynapticCompartmentName()] = x

    @property
    def presynSpike(self):
        return self.compartments.get(self.presynSpikeName(), None)

    @presynSpike.setter
    def presynSpike(self, x):
        self.compartments[self.presynSpikeName()] = x

    @property
    def postsynSpike(self):
        return self.compartments.get(self.postsynSpikeName(), None)

    @postsynSpike.setter
    def postsynSpike(self, x):
        self.compartments[self.postsynSpikeName()] = x

    # Define Functions
    def __init__(self, name, shape, eta=0., wInit=("uniform", 0., 0.3),
                 bInit=None, w_bound=1., is_nonnegative=False, w_decay=0.,
                 signVal=1., optim_type="sgd", key=None, useVerboseDict=False,
                 directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ## synaptic plasticity properties and characteristics
        self.shape = shape
        self.w_bounds = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.eta = eta
        self.wInit = wInit
        self.bInit = bInit
        self.is_nonnegative = is_nonnegative
        self.signVal = signVal

        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = None
        if optim_type == "adam":
            self.opt = Adam(learning_rate=self.eta)
        else: ## default is SGD
            self.opt = SGD(learning_rate=self.eta)

        if directory is None:
            self.key, subkey = random.split(self.key)
            self.weights = initialize_params(subkey, wInit, shape)
            if self.bInit is not None:
                self.key, subkey = random.split(self.key)
                self.biases = initialize_params(subkey, bInit, (1, shape[1]))
        else:
            self.load(directory)

        ##Reset to initialize stuff
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, **kwargs):
        biases = 0.
        if self.bInit != None:
            biases = self.biases
        self.outputCompartment = compute_layer(self.inputCompartment,
                                               self.weights, self.biases)

    def evolve(self, t, dt, **kwargs):
        dW, db = calc_update(self.presynapticCompartment, self.postsynapticCompartment,
                             self.weights, self.w_bounds, is_nonnegative=self.is_nonnegative,
                             signVal=self.signVal, w_decay=0.)
        if self.w_decay > 0.:
            dW = apply_decay(dW, self.presynSpike, self.postsynSpike, self.w_decay)

        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if self.bInit != None:
            theta = [self.weights, self.biases]
            self.opt.update(theta, [dW, db])
            self.weights = theta[0]
            self.biases = theta[1]
        else:
            theta = [self.weights]
            self.opt.update(theta, [dW])
            self.weights = theta[0]
        ## ensure synaptic efficacies adhere to constraints
        self.weights = enforce_constraints(self.weights, self.w_bounds,
                                           is_nonnegative=self.is_nonnegative)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = None
        self.presynapticCompartment = None
        self.postsynapticCompartment = None
        self.presynSpike = None
        self.postsynSpike = None

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights)
        if self.bInit != None:
            jnp.savez(file_name, biases=self.biases)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights = data['weights']
        if self.bInit != None:
            self.biases = data['biases']
