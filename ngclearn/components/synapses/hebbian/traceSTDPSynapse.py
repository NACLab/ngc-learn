from ngcsimlib.component import Component
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params, normalize_matrix
import time

@partial(jit, static_argnums=[6,7,8,9,10,11,12])
def evolve(dt, pre, x_pre, post, x_post, W, w_bound=1., eta=1.,
            x_tar=0.0, mu=0., Aplus=1., Aminus=0., w_norm=None):
    """
    Evolves/changes the synpatic value matrix underlying this synaptic cable,
    given relevant statistics.

    Args:
        pre: pre-synaptic statistic to drive update

        x_pre: pre-synaptic trace value

        post: post-synaptic statistic to drive update

        x_post: post-synaptic trace value

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        eta: global learning rate to apply to the Hebbian update

        x_tar: controls degree of pre-synaptic disconnect

        mu: controls the power scale of the Hebbian shift

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        w_norm: (Unused)

    Returns:
        the newly evolved synaptic weight value matrix
    """
    if mu > 0.:
        ## equations 3, 5, & 6 from Diehl and Cook - full power-law STDP
        post_shift = jnp.power(w_bound - W, mu)
        pre_shift = jnp.power(W, mu)
        dWpost = (post_shift * jnp.matmul((x_pre - x_tar).T, post)) * Aplus
        if Aminus > 0.:
            dWpre = -(pre_shift * jnp.matmul(pre.T, x_post)) * Aminus
    else:
        ## calculate post-synaptic term
        dWpost = jnp.matmul((x_pre - x_tar).T, post * Aplus)
        dWpre = 0.
        if Aminus > 0.:
            ## calculate pre-synaptic term
            dWpre = -jnp.matmul(pre.T, x_post * Aminus)
    ## calc final weighted adjustment
    dW = (dWpost + dWpre) * eta
    _W = W + dW
    # if w_norm is not None:
    #     _W = normalize_matrix(_W, w_norm, order=1, axis=1) ## L1 norm constraint
    #    #_W = _W * (w_norm/(jnp.linalg.norm(_W, axis=1, keepdims=True) + 1e-5))
    _W = jnp.clip(_W, 0.001, w_bound) # 0.01, w_bound)
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

class TraceSTDPSynapse(Component): # power-law / trace-based STDP
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP), including an optional power-scale
    dependence that can be equipped to the Hebbian adjustment (the strength of
    which is controlled by a scalar factor).

    | References:
    | Morrison, Abigail, Ad Aertsen, and Markus Diesmann. "Spike-timing-dependent
    | plasticity in balanced random networks." Neural computation 19.6 (2007): 1437-1467.
    |
    | Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modification by correlated
    | activity: Hebb's postulate revisited." Annual review of neuroscience 24.1
    | (2001): 139-166.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        mu: controls the power scale of the Hebbian shift

        preTrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        w_norm: if not None, applies an L1 norm constraint to synapses

        norm_T: clocked time at which to apply L1 synaptic norm constraint

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
    def presynapticTraceName(cls):
        return 'x_pre'

    @classmethod
    def postsynapticTraceName(cls):
        return 'x_post'

    @classmethod
    def triggerName(cls):
        return 'trigger'

    ## Bind Properties to Compartments for ease of use
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
    def trigger(self):
        return self.compartments.get(self.triggerName(), None)

    @trigger.setter
    def trigger(self, x):
        # FIXME: place a check in here? (should check for single float value)
        self.compartments[self.triggerName()] = x

    @property
    def presynapticTrace(self):
        return self.compartments.get(self.presynapticTraceName(), None)

    @presynapticTrace.setter
    def presynapticTrace(self, x):
        self.compartments[self.presynapticTraceName()] = x

    @property
    def postsynapticTrace(self):
        return self.compartments.get(self.postsynapticTraceName(), None)

    @postsynapticTrace.setter
    def postsynapticTrace(self, x):
        self.compartments[self.postsynapticTraceName()] = x

    # Define Functions
    def __init__(self, name, shape, eta, Aplus, Aminus, mu=0.,
                 preTrace_target=0., wInit=("uniform", 0.025, 0.8), w_norm=None,
                 norm_T=250., key=None, useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##parms
        self.shape = shape ## shape of synaptic efficacy matrix
        self.eta = eta ## global learning rate governing plasticity
        self.mu = mu ## controls power-scaling of STDP rule
        self.preTrace_target = preTrace_target ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = Aplus ## LTP strength
        self.Aminus = Aminus ## LTD strength
        self.shape = shape  # shape of synaptic matrix W
        self.w_bound = 1. ## soft weight constraint
        self.w_norm = w_norm ## normalization constant for synaptic matrix after update
        self.norm_T = norm_T ## scheduling time / checkpoint for synaptic normalization

        if directory is None:
            self.key, subkey = random.split(self.key)
            #self.weights = random.uniform(subkey, shape, minval=lb, maxval=ub)
            self.weights = initialize_params(subkey, wInit, shape)
        else:
            self.load(directory)

        ##Reset to initialize core compartments
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, dt, t, **kwargs):
        ## run signals across synapses
        self.outputCompartment = compute_layer(self.inputCompartment, self.weights)

    def evolve(self, dt, t, **kwargs):
        #trigger = self.trigger
        pre = self.inputCompartment
        post = self.outputCompartment
        x_pre = self.presynapticTrace
        x_post = self.postsynapticTrace
        self.weights = evolve(dt, pre, x_pre, post, x_post, self.weights,
                              w_bound=self.w_bound, eta=self.eta,
                              x_tar=self.preTrace_target, mu=self.mu,
                              Aplus=self.Aplus, Aminus=self.Aminus,
                              w_norm=self.w_norm)
        if self.norm_T > 0:
            if t % (self.norm_T-1) == 0: #t % self.norm_t == 0:
                self.weights = normalize_matrix(self.weights, self.w_norm, order=1, axis=0)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = None
        self.presynapticTrace = None
        self.postsynapticTrace = None
        self.trigger = 1. ## default: assume synaptic change will occur

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights = data['weights']
