from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
import jax
from typing import Union

class BernoulliCell(JaxComponent):
    """
    A Bernoulli cell that produces spikes by sampling a Bernoulli distribution
    on-the-fly (to produce data-scaled Bernoulli spike trains).

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals -- should be probabilities w/ values in [0,1])
    | --- Cell State Compartments: ---
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | outputs - output
    | tols - time-of-last-spike

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        batch_size: batch size dimension of this cell (Default: 1)
    """

    def __init__(self, name: str, n_units: int, batch_size: int = 1, key: Union[jax.Array, None] = None, **kwargs):
        super().__init__(name=name, key=key)

        ## Layer Size Setup
        self.batch_size = Compartment(batch_size)
        self.n_units = Compartment(n_units)

        restVals = jnp.zeros((batch_size, n_units))
        self.inputs = Compartment(restVals, display_name="Input Stimulus") # input compartment
        self.outputs = Compartment(restVals, display_name="Spikes") # output compartment
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike", units="ms") # time of last spike

    @compilable
    def advance_state(self, t):
        key, subkey = random.split(self.key.get(), 2)
        self.outputs.set(random.bernoulli(subkey, p=self.inputs.get()).astype(jnp.float32))
        self.tols.set((1. - self.outputs.get()) * self.tols.get() + (self.outputs.get() * t))
        self.key.set(key)

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size.get(), self.n_units.get()))
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(restVals)
        self.outputs.set(restVals)
        self.tols.set(restVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "BernoulliCell - samples input to produce spikes, "
                          "where dimension is a probability proportional to "
                          "the dimension's magnitude/value/intensity"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"key": "JAX PRNG key"},
            "outputs":
                {"tols": "Time-of-last-spike",
                 "outputs": "Binary spike values emitted at time t"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "~ Bernoulli(x)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = BernoulliCell("X", 9)

    X.batch_size.set(10)
