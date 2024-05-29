from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from ngclearn.utils import tensorstats

from jax import numpy as jnp
import time, sys

class TraceSTDPSynapse(Component): ## Lava-compliant trace-STDP synapse

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
        self.eta = Compartment(jnp.ones((1,1)) * eta)

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
        weights = weights + dW * eta - weights * w_decay
        #weights = weights + (dW - weights * w_decay) * dt/tau_w ## ODE format
        weights = jnp.clip(weights, 0., w_bounds)
        ########################################################################
        return outputs, weights

    @resolver(_advance_state)
    def advance_state(self, outputs, weights):
        self.outputs.set(outputs)
        self.weights.set(weights)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            preVals, # x_pre
            postVals # x_post
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post, x_pre, x_post):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
        self.x_pre.set(x_pre)
        self.x_post.set(x_post)
        #self.eta.set(jnp.ones((1,1)) * eta0)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set(data['weights'])

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
