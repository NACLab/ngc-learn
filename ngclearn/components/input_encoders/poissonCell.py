from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
from ngcsimlib.logger import info, warn

from ngcsimlib.compilers.process import transition
#from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

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
    def __init__(self, name, n_units, target_freq=63.75, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

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

    def validate(self, dt=None, **validation_kwargs):
        valid = super().validate(**validation_kwargs)
        if dt is None:
            warn(f"{self.name} requires a validation kwarg of `dt`")
            return False
        ## check for unstable combinations of dt and target-frequency meta-params
        events_per_timestep = (dt/1000.) * self.target_freq ## compute scaled probability
        if events_per_timestep > 1.:
            valid = False
            warn(
                f"{self.name} will be unable to make as many temporal events as "
                f"requested! ({events_per_timestep} events/timestep) Unstable "
                f"combination of dt = {dt} and target_freq = {self.target_freq} "
                f"being used!"
            )
        return valid

    @transition(output_compartments=["outputs", "tols", "key"])
    @staticmethod
    def advance_state(t, dt, target_freq, key, inputs, tols):
        key, *subkeys = random.split(key, 2)
        pspike = inputs * (dt / 1000.) * target_freq
        eps = random.uniform(subkeys[0], inputs.shape, minval=0., maxval=1.,
                             dtype=jnp.float32)
        outputs = (eps < pspike).astype(jnp.float32)

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
        target_freq = (self.target_freq if isinstance(self.target_freq, float)
                       else jnp.ones([[self.target_freq]]))
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, key=self.key.value, target_freq=target_freq)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.key.set(data['key'])
        self.target_freq = data['target_freq']

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

    def __repr__(self):
        comps = [varname for varname in dir(self) if
                 Compartment.is_compartment(getattr(self, varname))]
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
