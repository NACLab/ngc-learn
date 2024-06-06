from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from ngclearn.utils import tensorstats

@jit
def update_times(t, s, tols):
    """
    Updates time-of-last-spike (tols) variable.

    Args:
        t: current time (a scalar/int value)

        s: binary spike vector

        tols: current time-of-last-spike variable

    Returns:
        updated tols variable
    """
    _tols = (1. - s) * tols + (s * t)
    return _tols

@jit
def sample_bernoulli(dkey, data):
    """
    Samples a Bernoulli spike train on-the-fly

    Args:
        dkey: JAX key to drive stochasticity/noise

        data: sensory data (vector/matrix)

    Returns:
        binary spikes
    """
    s_t = random.bernoulli(dkey, p=data).astype(jnp.float32)
    return s_t

class BernoulliCell(JaxComponent):
    """
    A Bernoulli cell that produces Bernoulli-distributed spikes on-the-fly.

    | --- Cell Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output
    | tols - time-of-last-spike
    | key - JAX RNG key

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)
    """

    # Define Functions
    def __init__(self, name, n_units, key=None, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        # Compartments (state of the cell, parameters, will be updated through stateless calls)
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input compartment
        self.outputs = Compartment(restVals) # output compartment
        self.tols = Compartment(restVals) # time of last spike

    @staticmethod
    def _advance_state(t, dt, key, inputs, tols):
        key, *subkeys = random.split(key, 2)
        outputs = sample_bernoulli(subkeys[0], data=inputs)
        timeOfLastSpike = update_times(t, outputs, tols)
        return outputs, timeOfLastSpike, key

    @resolver(_advance_state)
    def advance_state(self, outputs, tols, key):
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.key.set(key)

    @staticmethod
    def _reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        return restVals, restVals, restVals

    @resolver(_reset)
    def reset(self, inputs, outputs, tols):
        self.inputs.set(inputs)
        self.outputs.set(outputs) #None
        self.tols.set(tols)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, key=self.key.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.key.set( data['key'] )

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
