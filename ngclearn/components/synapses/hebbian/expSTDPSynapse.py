from ngcsimlib.component import Component
from jax import random, numpy as jnp, jit
from functools import partial
import time

@partial(jit, static_argnums=[6,7,8,9,10,11,12])
def evolve(dt, pre, x_pre, post, x_post, W, w_bound=1., eta=0.00005,
            x_tar=0.7, exp_beta=1., Aplus=1., Aminus=0., w_norm=None):
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

        exp_beta: controls effect of exponential Hebbian shift/dependency

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        w_norm: if not None, applies an L2 norm constraint to synapses

    Returns:
        the newly evolved synaptic weight value matrix
    """
    ## equations 4 from Diehl and Cook - full exponential weight-dependent STDP
    ## calculate post-synaptic term
    post_term1 = jnp.exp(-exp_beta * W) * jnp.matmul(x_pre.T, post)
    x_tar_vec = x_pre * 0 + x_tar # need to broadcast scalar x_tar to mat/vec form
    post_term2 = jnp.exp(-exp_beta * (w_bound - W)) * jnp.matmul(x_tar_vec.T, post)
    dWpost = (post_term1 - post_term2) * Aplus
    ## calculate pre-synaptic term
    dWpre = 0.
    if Aminus > 0.:
        dWpre = -jnp.exp(-exp_beta * W) * jnp.matmul(pre.T, x_post) * Aminus

    ## calc final weighted adjustment
    dW = (dWpost + dWpre) * eta
    _W = W + dW
    if w_norm is not None:
        _W = _W * (w_norm/(jnp.linalg.norm(_W, axis=1, keepdims=True) + 1e-5))
    _W = jnp.clip(_W, 0.01, w_bound) # not in source paper
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

class ExpSTDPSynapse(Component):
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP) based on an exponential weight
    dependence (the strength of which is controlled by a factor).

    | References:
    | Nessler, Bernhard, et al. "Bayesian computation emerges in generic cortical
    | microcircuits through spike-timing-dependent plasticity." PLoS computational
    | biology 9.4 (2013): e1003037.
    |
    | Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modification by correlated
    | activity: Hebb's postulate revisited." Annual review of neuroscience 24.1
    | (2001): 139-166.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate

        exp_beta: controls effect of exponential Hebbian shift/dependency

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        preTrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

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
    def __init__(self, name, shape, eta, exp_beta, Aplus, Aminus,
                 preTrace_target, wInit=(0.025, 0.8), key=None, useVerboseDict=False,
                 directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##parms
        self.shape = shape ## shape of synaptic efficacy matrix
        self.eta = eta ## global learning rate governing plasticity
        self.exp_beta = exp_beta ## if not None, will trigger exp-depend STPD rule
        self.preTrace_target = preTrace_target ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = Aplus ## LTP strength
        self.Aminus = Aminus ## LTD strength
        self.shape = shape  # shape of synaptic matrix W
        self.w_bound = 1. ## soft weight constraint
        self.w_norm = None ## normalization constant for synaptic matrix after update

        if directory is None:
            self.key, subkey = random.split(self.key)
            lb, ub = wInit
            self.weights = random.uniform(subkey, shape, minval=lb, maxval=ub)
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
        pre = self.inputCompartment
        post = self.outputCompartment
        x_pre = self.presynapticTrace
        x_post = self.postsynapticTrace
        self.weights = evolve(dt, pre, x_pre, post, x_post, self.weights,
                              w_bound=self.w_bound, eta=self.eta,
                              x_tar=self.preTrace_target, exp_beta=self.exp_beta,
                              Aplus=self.Aplus, Aminus=self.Aminus)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = None
        self.presynapticTrace = None
        self.postsynapticTrace = None

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights = data['weights']
