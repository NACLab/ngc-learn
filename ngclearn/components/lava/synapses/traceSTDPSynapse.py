from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats

from ngclearn import numpy as jnp
import time, sys

class TraceSTDPSynapse(Component): ## Lava-compliant trace-STDP synapse
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP). This is a Lava-compliant synaptic
    cable that adjusts with a hard-coded form of (stochastic) gradient ascent.

    Args:
        name: the string name of this cell

        weights: matrix of synaptic weight values to initialize this synapse
            component to

        dt: integration time constant (ms)

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        eta: global learning rate

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed Hebbian adjustment (Default: 0); note that decay is not
            applied to any configured biases

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        preTrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)
    """

    # Define Functions
    def __init__(self, name, weights, dt, Rscale=1., Aplus=0.01, Aminus=0.001,
                 eta=1., w_decay=0., w_bound=1., preTrace_target=0., **kwargs):
        super().__init__(name, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.dt = dt
        self.Rscale = Rscale
        self.w_bounds = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.eta0 = eta
        self.Aplus = Aplus
        self.Aminus = Aminus
        self.x_tar = preTrace_target

        ## Component size setup
        self.batch_size = 1
        self.shape = None

        ## Compartments
        self.inputs = Compartment(None)
        self.outputs = Compartment(None)
        self.pre = Compartment(None) ## pre-synaptic spike
        self.x_pre = Compartment(None) ## pre-synaptic trace
        self.post = Compartment(None) ## post-synaptic spike
        self.x_post = Compartment(None) ## post-synaptic trace
        self.weights = Compartment(None)
        self.eta = Compartment(jnp.ones((1,1)) * eta)

        if weights is not None:
            self._init(weights)

    def _init(self, weights):
        self.shape = weights.shape
        ## pre-computed empty zero pads
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        ## Compartments
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.pre = Compartment(preVals) ## pre-synaptic spike
        self.x_pre = Compartment(preVals) ## pre-synaptic trace
        self.post = Compartment(postVals) ## post-synaptic spike
        self.x_post = Compartment(postVals) ## post-synaptic trace
        self.weights = Compartment(weights)

    @staticmethod
    def _advance_state(dt, Rscale, Aplus, Aminus, w_bounds, w_decay, x_tar,
                       inputs, weights, pre, x_pre, post, x_post, eta):
        outputs = jnp.matmul(inputs, weights) * Rscale
        ########################################################################
        ## Run one step of STDP online
        dWpost = jnp.matmul((x_pre - x_tar).T, post * Aplus)
        dWpre = -jnp.matmul(pre.T, x_post * Aminus)
        dW = dWpost + dWpre
        ## reformulated bounding flag to be linear algebraic
        flag = (w_bounds > 0.) * 1.
        dW = (dW * (w_bounds - jnp.abs(weights))) * flag + (dW) * (1. - flag)
        ## physically adjust synapses
        weights = weights + (dW - weights * w_decay) * eta
        #weights = weights + (dW - weights * w_decay) * dt/tau_w ## ODE format
        weights = jnp.clip(weights, 0., w_bounds)
        ########################################################################
        return outputs, weights

    @resolver(_advance_state)
    def advance_state(self, outputs, weights):
        self.outputs.set(outputs)
        self.weights.set(weights)

    @staticmethod
    def _reset(batch_size, shape, eta0):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            preVals, # x_pre
            postVals, # x_post
            jnp.ones((1,1)) * eta0
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post, x_pre, x_post, eta):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
        self.x_pre.set(x_pre)
        self.x_post.set(x_post)
        self.eta.set(eta)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self._init( data['weights'] )
