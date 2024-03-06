from ngclib.component import Component
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params
import time

@partial(jit, static_argnums=[3])
def update_eligibility(dt, Eg, dW, elg_tau):
    _Eg = Eg + (-Eg + dW) * (dt/elg_tau)
    return _Eg

@partial(jit, static_argnums=[3,4,5])
def _calc_update(pre, post, W, w_bound, is_nonnegative=True, signVal=1.):
    dW = jnp.matmul(pre.T, post)
    if w_bound > 0.:
        dW = dW * (w_bound - jnp.abs(W))
    return dW * signVal

@partial(jit, static_argnums=[2,3,4])
def _adjust_synapses(dW, W, w_bound, eta, is_nonnegative=True):
    _W = W + dW * eta
    if w_bound > 0.:
        if is_nonnegative == True:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W

@partial(jit, static_argnums=[4,5])
def _evolve(pre, post, W, w_bound, eta, is_nonnegative=True):
    dW = _calc_update(pre, post, W, w_bound, is_nonnegative)
    _W = _adjust_synapses(dW, W, w_bound, eta, is_nonnegative)
    return _W

@jit
def _compute_layer(inp, weight):
    return jnp.matmul(inp, weight)

class HebbianSynapse(Component):
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
        # if x is not None:
        #     if type(x) == float:
        #         raise RuntimeError(
        #             "Trigger compartment must be a single float and provided " +
        #             "argument type " + type(x) + " does not match this type.")
        self.compartments[self.triggerName()] = x

    @property
    def inputCompartment(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, x):
        if x is not None:
            if x.shape[1] != self.shape[0]:
                raise RuntimeError(
                    "Input compartment size (n, " + str(self.shape[0]) + ") does not match provided input size "
                    + str(x.shape) + " for " + str(self.name))
        self.compartments[self.inputCompartmentName()] = x

    @property
    def outputCompartment(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @outputCompartment.setter
    def outputCompartment(self, x):
        if x is not None:
            if x.shape[1] != self.shape[1]:
                raise RuntimeError(
                    "Output compartment size (n, " + str(self.shape[1]) + ") does not match provided output size "
                    + str(x.shape) + " for " + str(self.name))
        self.compartments[self.outputCompartmentName()] = x

    @property
    def presynapticCompartment(self):
        return self.compartments.get(self.presynapticCompartmentName(), None)

    @presynapticCompartment.setter
    def presynapticCompartment(self, x):
        if x is not None:
            if x.shape[1] != self.shape[0]:
                raise RuntimeError(
                    "Presynaptic compartment size (n, " + str(
                        self.shape[0]) + ") does not match provided presynaptic size "
                    + str(x.shape) + " for " + str(self.name))
        self.compartments[self.presynapticCompartmentName()] = x

    @property
    def postsynapticCompartment(self):
        return self.compartments.get(self.postsynapticCompartmentName(), None)

    @postsynapticCompartment.setter
    def postsynapticCompartment(self, x):
        if x is not None:
            if x.shape[1] != self.shape[1]:
                raise RuntimeError(
                    "Postsynaptic compartment size (n, " + str(
                        self.shape[1]) + ") does not match provided postsynaptic size "
                    + str(x.shape) + " for " + str(self.name))
        self.compartments[self.postsynapticCompartmentName()] = x

    # Define Functions
    def __init__(self, name, shape, eta, wInit=("uniform", 0., 0.3), w_bound=1.,
                 elg_tau=0., is_nonnegative=False, signVal=1., key=None,
                 useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##params
        self.shape = shape
        self.w_bounds = w_bound
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
        self.outputCompartment = _compute_layer(self.inputCompartment, self.weights)

    def evolve(self, t, dt, **kwargs):
        if self.elg_tau > 0.:
            trigger = self.trigger
            dW = _calc_update(self.presynapticCompartment, self.postsynapticCompartment,
                              self.weights, self.w_bounds, is_nonnegative=self.is_nonnegative,
                              signVal=self.signVal)
            self.Eg = update_eligibility(dt, self.Eg, dW, self.elg_tau)
            if trigger > 0.:
                self.weights = _adjust_synapses(self.Eg, self.weights, self.w_bounds, self.eta,
                                                is_nonnegative=self.is_nonnegative)
        else:
            dW = _calc_update(self.presynapticCompartment, self.postsynapticCompartment,
                              self.weights, self.w_bounds, is_nonnegative=self.is_nonnegative,
                              signVal=self.signVal)
            self.Eg = dW
            self.weights = _adjust_synapses(dW, self.weights, self.w_bounds, self.eta,
                                            is_nonnegative=self.is_nonnegative)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = None
        self.presynapticCompartment = None
        self.postsynapticCompartment = None
        self.Eg = self.weights * 0

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights = data['weights']
