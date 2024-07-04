from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from jax import numpy as jnp, random, jit
from functools import partial

@jit
def _update_times(t, s, tols):
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

@partial(jit, static_argnums=[3])
def _sample_poisson(dkey, data, dt, fmax=63.75):
    """
    Samples a Poisson spike train on-the-fly.

    Args:
        dkey: JAX key to drive stochasticity/noise

        data: sensory data (vector/matrix)

        dt: integration time constant

        fmax: maximum frequency (Hz)

    Returns:
        binary spikes
    """
    pspike = data * (dt/1000.) * fmax
    eps = random.uniform(dkey, data.shape, minval=0., maxval=1., dtype=jnp.float32)
    s_t = (eps < pspike).astype(jnp.float32)
    return s_t

class PoissonCell(JaxComponent):
    """
    A Poisson cell that produces approximately Poisson-distributed spikes on-the-fly.

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

        max_freq: maximum frequency (in Hertz) of this Poisson spike train (must be > 0.)
    """

    # Define Functions
    def __init__(self, name, n_units, max_freq=63.75, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        ## Poisson meta-parameters
        self.max_freq = max_freq ## maximum frequency (in Hertz/Hz)

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input compartment
        self.outputs = Compartment(restVals) # output compartment
        self.tols = Compartment(restVals) # time of last spike

    @staticmethod
    def _advance_state(t, dt, max_freq, key, inputs, tols):
        key, *subkeys = random.split(key, 2)
        outputs = _sample_poisson(subkeys[0], data=inputs, dt=dt, fmax=max_freq)
        tols = _update_times(t, outputs, tols)
        return outputs, tols, key

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
        self.outputs.set(outputs)
        self.tols.set(tols)

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
            "cell_type": "PoissonCell - samples input to produce spikes, "
                          "where dimension is a probability proportional to "
                          "the dimension's magnitude/value/intensity and "
                         "constrained by a maximum spike frequency (spikes follow "
                         "a Poisson distribution)"
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
            "batch_size": "Batch size dimension of this component",
            "max_freq": "Maximum spike frequency of the train produced",
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "~ Poisson(x; max_freq)",
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
        X = PoissonCell("X", 9)
    print(X)
