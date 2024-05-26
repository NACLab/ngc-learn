from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from ngclearn.utils import tensorstats

from jax import numpy as jnp, jit
from functools import partial
import time

class TraceSTDPSynapse(Component): ## Lava-compliant Hebbian synapse

    # Define Functions
    def __init__(self, name, weights, Rscale=1., Aplus=0.01, Aminus=0.001,
                 w_decay=0., w_bound=1., preTrace_target=0., **kwargs):
        super().__init__(name, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.batch_size = 1
        self.shape = weights.shape
        self.Rscale = Rscale
        self.w_bounds = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.Aplus = Aplus
        self.Aminus = Aminus
        self.x_tar = preTrace_target

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
    def _advance_state(t, dt, Rscale, inputs, weights):
        outputs = jnp.matmul(inputs, weights) * Rscale
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _evolve(t, dt, Aplus, Aminus, w_bounds, w_decay, x_tar, pre, x_pre,
                post, x_post, weights):
        dWpost = jnp.matmul((x_pre - x_tar).T, post * Aplus)
        dWpre = -jnp.matmul(pre.T, x_post * Aminus)
        dW = dWpost + dWpre
        ## reformulated bounding flag to be linear algebraic
        flag = (w_bounds > 0.) * 1.
        dW = (dW * (w_bounds - jnp.abs(weights))) * flag + (dW) * (1. - flag)
        ## add small amount of synaptic decay
        dW = dW - weights * w_decay
        ## physically adjust synapses
        weights = weights + dW * eta
        #weights = weights + (dW - weights * w_decay) * dt/tau_w
        #weights = jnp.clip(weights, 0., w_bounds)
        return weights

    @resolver(_evolve)
    def evolve(self, weights):
        self.weights.set(weights)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals # post
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)

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
