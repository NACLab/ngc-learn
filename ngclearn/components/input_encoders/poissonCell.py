from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from jax import numpy as jnp, random, jit, scipy
from functools import partial
from ngcsimlib.deprecators import deprecate_args
from ngcsimlib.logger import info, warn

class PoissonCell(JaxComponent):
    """
    A Poisson cell that produces approximately Poisson-distributed spikes
    on-the-fly.

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

        max_freq: maximum frequency (in Hertz) of this Poisson spike train (
        must be > 0.)
    """

    # Define Functions
    @deprecate_args(max_freq="target_freq")
    def __init__(self, name, n_units, target_freq=63.75, batch_size=1,
                 **kwargs):
        super().__init__(name, **kwargs)

        ## Poisson meta-parameters
        self.target_freq = target_freq  ## maximum frequency (in Hertz/Hz)

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        _key, subkey = random.split(self.key.value, 2)
        self.key.set(_key)
        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals,
                                  display_name="Input Stimulus")  # input
        # compartment
        self.outputs = Compartment(restVals,
                                   display_name="Spikes")  # output compartment
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike",
                                units="ms")  # time of last spike
        self.targets = Compartment(
            random.uniform(subkey, (self.batch_size, self.n_units), minval=0.,
                           maxval=1.))

    def validate(self, dt, **validation_kwargs):
        ## check for unstable combinations of dt and target-frequency meta-params
        valid = super().validate(**validation_kwargs)
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

    @staticmethod
    def _advance_state(t, dt, target_freq, key, inputs, targets, tols):
        ms_per_second = 1000  # ms/s
        events_per_ms = target_freq / ms_per_second  # e/s s/ms -> e/ms
        ms_per_event = 1 / events_per_ms  # ms/e
        time_step_per_event = ms_per_event / dt  # ms/e * ts/ms -> ts / e

        cdf = scipy.special.gammaincc((t + dt) - tols,
                                      time_step_per_event/inputs)
        outputs = (targets < cdf).astype(jnp.float32)

        key, subkey = random.split(key, 2)
        targets = (targets * (1 - outputs) + random.uniform(subkey,
                                                           targets.shape) *
                   outputs)

        tols = tols * (1. - outputs) + t * outputs
        return outputs, tols, key, targets

    @resolver(_advance_state)
    def advance_state(self, outputs, tols, key, targets):
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.key.set(key)
        self.targets.set(targets)

    @staticmethod
    def _reset(batch_size, n_units, key):
        restVals = jnp.zeros((batch_size, n_units))
        key, subkey = random.split(key, 2)
        targets = random.uniform(subkey, (batch_size, n_units))
        return restVals, restVals, restVals, targets, key

    @resolver(_reset)
    def reset(self, inputs, outputs, tols, targets, key):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.key.set(key)
        self.targets.set(targets)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, key=self.key.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.key.set(data['key'])

    @classmethod
    def help(cls):  ## component help function
        properties = {
            "cell_type": "PoissonCell - samples input to produce spikes, "
                         "where dimension is a probability proportional to "
                         "the dimension's magnitude/value/intensity and "
                         "constrained by a maximum spike frequency (spikes "
                         "follow "
                         "a Poisson distribution)"
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
