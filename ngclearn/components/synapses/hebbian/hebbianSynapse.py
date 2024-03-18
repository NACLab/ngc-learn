from ngclib.component import Component
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params
import time

@partial(jit, static_argnums=[3])
def update_eligibility(dt, Eg, dW, elg_tau):
    """
    Apply a signal modulator to j (typically of the form of a derivative/dampening function)

    Args:
        dt: integration time constant

        Eg: current value of the eligibility trace

        dW: synaptic update (at t) to update eligibility trace with

        elg_tau: eligibility trace time constant

    Returns:
        updated eligibility trace tensor Eg
    """
    _Eg = Eg + (-Eg + dW) * (dt/elg_tau)
    return _Eg

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
        an update/adjustment matrix
    """
    dW = jnp.matmul(pre.T, post)
    if w_bound > 0.:
        dW = dW * (w_bound - jnp.abs(W))
    if w_decay > 0.:
        dW = dW - W * w_decay # jnp.matmul((1. - pre).T, (1. - post)) * w_decay
    return dW * signVal

@partial(jit, static_argnums=[2,3,4])
def adjust_synapses(dW, W, w_bound, eta, is_nonnegative=True):
    """
    Evolves/changes the synpatic value matrix underlying this synaptic cable,
    given a computed synaptic update.

    Args:
        dW: synaptic adjustment matrix to be applied/used

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        eta: global learning rate to apply to the Hebbian update

        is_nonnegative: ensure updated value matrix is strictly non-negative

    Returns:
        the newly evolved synaptic weight value matrix
    """
    _W = W + dW * eta
    if w_bound > 0.:
        if is_nonnegative == True:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W

#@jit
def apply_decay(dW, pre_s, post_s, w_decay):
    _dW = dW - jnp.matmul((1. - pre_s).T, (1. - post_s)) * w_decay
    sys.exit(0)
    return _dW

@partial(jit, static_argnums=[4,5])
def evolve(pre, post, W, w_bound, eta, is_nonnegative=True):
    """
    Evolves/changes the synpatic value matrix underlying this synaptic cable,
    given relevant statistics.

    Args:
        pre: pre-synaptic statistic to drive Hebbian update

        post: post-synaptic statistic to drive Hebbian update

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        eta: global learning rate to apply to the Hebbian update

        is_nonnegative: ensure updated value matrix is strictly non-negative

    Returns:
        the newly evolved synaptic weight value matrix
    """
    dW = calc_update(pre, post, W, w_bound, is_nonnegative)
    _W = adjust_synapses(dW, W, w_bound, eta, is_nonnegative)
    return _W

@jit
def compute_layer(inp, weight):
    """
    Applies the transformation/projection induced by the synaptic efficacie
    associated with this synaptic cable

    Args:
        inp: signal input to run through this synaptic cable

        weight: this cable's synaptic value matrix

    Returns:
        a projection/transformation of input "inp"
    """
    return jnp.matmul(inp, weight)

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

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        elg_tau: if > 0., triggers the use of an eligibility trace where this value
            serves as its time constant

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)

        signVal: multiplicative factor to apply to final synaptic update before
            it is applied to synapses; this is useful if gradient descent schemes
            are to be applied (as Hebbian rules typically yield adjustments for
            ascent)

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
    def __init__(self, name, shape, eta, wInit=("uniform", 0., 0.3), w_bound=1.,
                 elg_tau=0., is_nonnegative=False, w_decay=0., signVal=1., key=None,
                 useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##params
        self.shape = shape
        self.w_bounds = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.eta = eta
        self.wInit = wInit
        self.is_nonnegative = is_nonnegative
        self.signVal = signVal
        self.elg_tau = elg_tau
        self.Eg = None

        if directory is None:
            self.key, subkey = random.split(self.key)
            self.weights = initialize_params(subkey, wInit, shape)
        else:
            self.load(directory)

        ##Reset to initialize stuff
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, **kwargs):
        self.outputCompartment = compute_layer(self.inputCompartment, self.weights)

    def evolve(self, t, dt, **kwargs):
        if self.elg_tau > 0.:
            trigger = self.trigger
            dW = calc_update(self.presynapticCompartment, self.postsynapticCompartment,
                             self.weights, self.w_bounds, is_nonnegative=self.is_nonnegative,
                             signVal=self.signVal)
            self.Eg = update_eligibility(dt, self.Eg, dW, self.elg_tau)
            if trigger > 0.:
                self.weights = adjust_synapses(self.Eg, self.weights, self.w_bounds, self.eta,
                                               is_nonnegative=self.is_nonnegative)
        else:
            dW = calc_update(self.presynapticCompartment, self.postsynapticCompartment,
                             self.weights, self.w_bounds, is_nonnegative=self.is_nonnegative,
                             signVal=self.signVal, w_decay=0.)
            if self.w_decay > 0.:
                dW = apply_decay(dW, self.presynSpike, self.postsynSpike, self.w_decay)
            self.Eg = dW
            self.weights = adjust_synapses(dW, self.weights, self.w_bounds, self.eta,
                                           is_nonnegative=self.is_nonnegative)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = None
        self.presynapticCompartment = None
        self.postsynapticCompartment = None
        self.presynSpike = None
        self.postsynSpike = None
        self.Eg = self.weights * 0

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights = data['weights']
