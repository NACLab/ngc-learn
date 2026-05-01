from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random
import jax
from typing import Union

from ngcsimlib import deprecate_args
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

class PoissonCell(JaxComponent):
    """
    A Poisson cell that samples a homogeneous Poisson process on-the-fly to
    produce a spike train.

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell State Compartments: ---
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | outputs - output
    | tols - time-of-last-spike

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        target_freq: maximum frequency (in Hertz) of this Bernoulli spike train (must be > 0.)

        batch_size: batch size dimension of this cell (Default: 1)
    """

    @deprecate_args(max_freq="target_freq")
    def __init__(
            self, name: str, n_units: int, target_freq: float = 63.75, batch_size: int = 1,
            key: Union[jax.Array, None] = None, **kwargs
    ):
        super().__init__(name=name, key=key)

        ## Constrained Bernoulli meta-parameters
        self.target_freq = target_freq  ## maximum frequency (in Hertz/Hz)

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        # Compartments (state of the cell, parameters, will be updated through stateless calls)
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals, display_name="Input Stimulus") # input compartment
        self.outputs = Compartment(restVals, display_name="Spikes") # output compartment
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike", units="ms") # time of last spike

    @compilable
    def advance_state(self, t, dt):
        key, subkey = random.split(self.key.get(), 2)
        pspike = self.inputs.get() * (dt / 1000.) * self.target_freq
        eps = random.uniform(subkey, self.inputs.get().shape, minval=0., maxval=1.,
                             dtype=jnp.float32)

        self.outputs.set((eps < pspike).astype(jnp.float32))
        self.tols.set((1. - self.outputs.get()) * self.tols.get() + (self.outputs.get() * t))
        self.key.set(key)

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size, self.n_units))
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(restVals)
        self.outputs.set(restVals)
        self.tols.set(restVals)

    @classmethod
    def help(cls):  ## component help function
        properties = {
            "cell_type": "PoissonCell - samples input to produce spikes, where dimension is a probability proportional "
                         "to the dimension's magnitude/value/intensity and constrained by a maximum spike frequency "
                         "(spikes follow a Poisson distribution)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"key": "JAX PRNG key",
                 "targets": "Target cdf for the Poisson distribution"},
            "outputs":
                {"tols": "Time-of-last-spike",
                 "outputs": "Binary spike values emitted at time t"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "target_freq": "Maximum spike frequency of the train produced",
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "~ Poisson(x; target_freq)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context

    with Context("Bar") as bar:
        X = PoissonCell("X", 9)
    print(X)
