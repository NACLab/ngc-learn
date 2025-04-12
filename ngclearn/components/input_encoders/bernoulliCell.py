from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
from ngcsimlib.logger import info, warn

from ngcsimlib.compilers.process import transition
#from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

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

    def __init__(self, name, n_units, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)
        #super(BernoulliCell, self).__init__(name, **kwargs)

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        # Compartments (state of the cell, parameters, will be updated through stateless calls)
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals, display_name="Input Stimulus") # input compartment
        self.outputs = Compartment(restVals, display_name="Spikes") # output compartment
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike", units="ms") # time of last spike

    @transition(output_compartments=["outputs", "tols", "key"])
    @staticmethod
    def advance_state(t, key, inputs, tols):
        ## NOTE: should `inputs` be checked if bounded to [0,1]?
        # print(key)
        # print(t)
        # print(inputs.shape)
        # print(tols.shape)
        # print("-----")
        key, *subkeys = random.split(key, 3)
        outputs = random.bernoulli(subkeys[0], p=inputs).astype(jnp.float32)
        # Updates time-of-last-spike (tols) variable:
        # output = s = binary spike vector
        # tols = current time-of-last-spike variable
        tols = (1. - outputs) * tols + (outputs * t)
        return outputs, tols, key

    @transition(output_compartments=["inputs", "outputs", "tols"])
    @staticmethod
    def reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        return restVals, restVals, restVals

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, key=self.key.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.key.set(data['key'])

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

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = BernoulliCell("X", 9)
    print(X)
