from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats

from ngclearn import numpy as jnp
import time

class StaticSynapse(Component): ## Lava-compliant fixed/non-evolvable synapse
    """
    A static (dense) synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component. This a Lava-compliant version of the
    static synapse component from the synapses sub-package of components.

    Args:
        name: the string name of this cell

        weights: matrix of synaptic weight values to initialize this synapse
            component to

        dt: integration time constant (ms)

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b
    """

    # Define Functions
    def __init__(self, name, weights, dt, Rscale=1., **kwargs):
        super().__init__(name, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.batch_size = 1
        self.dt = dt
        self.Rscale = Rscale
        self.shape = None

        ## Compartments
        self.inputs = Compartment(None)
        self.outputs = Compartment(None)
        self.weights = Compartment(None)

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
        self.weights = Compartment(weights)

    @staticmethod
    def _advance_state(dt, Rscale, inputs, weights):
        outputs = jnp.matmul(inputs, weights) * Rscale
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
        )

    @resolver(_reset)
    def reset(self, inputs, outputs):
        self.inputs.set(inputs)
        self.outputs.set(outputs)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self._init( data['weights'] )

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
