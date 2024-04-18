from ngcsimlib.component import Component
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params
from ngclearn.utils.optim import SGD, Adam
import time

@partial(jit, static_argnums=[3,4,5,6,7,8])
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

    # Define Functions
    def __init__(self, name, shape, eta=0., wInit=("uniform", 0., 0.3),
                 bInit=None, w_bound=1., is_nonnegative=False, w_decay=0.,
                 signVal=1., optim_type="sgd", pre_wght=1., post_wght=1.,
                 Rscale=1., key=None, useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ## random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

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

        self.dW = None
        self.db = None

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, **kwargs):
        biases = 0.
        if self.bInit != None:
            biases = self.biases
        self.outputCompartment = compute_layer(self.inputCompartment, self.weights,
                                               biases, self.Rscale)

    def evolve(self, t, dt, **kwargs):
        dW, db = calc_update(self.presynapticCompartment, self.postsynapticCompartment,
                             self.weights, self.w_bounds, is_nonnegative=self.is_nonnegative,
                             signVal=self.signVal, w_decay=self.w_decay,
                             pre_wght=self.pre_wght, post_wght=self.post_wght)
        self.dW = dW
        self.db = db
        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if self.bInit != None:
            theta = [self.weights, self.biases]
            self.opt.update(theta, [dW, db])
            self.weights = theta[0]
            self.biases = theta[1]
        else:
            # ignore db since no biases configured
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
        self.dW = None
        self.db = None

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.bInit != None:
            jnp.savez(file_name, weights=self.weights, biases=self.biases)
        else:
            jnp.savez(file_name, weights=self.weights)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights = data['weights']
        if "biases" in data.keys():
            self.biases = data['biases']
