from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random
import jax
from typing import Union

from ngcsimlib.compartment import Compartment
from ngcsimlib.parser import compilable


class PhasorCell(JaxComponent):
    """
    A phasor cell that emits a pulse at a regular interval.

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell State Compartments: ---
    | key - JAX PRNG key
    | angles - current angle of phasor 
    | --- Cell Output Compartments: ---
    | outputs - output of phasor cell
    | tols - time-of-last-spike

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        target_freq: maximum frequency (in Hertz) of this spike train
            (must be > 0.)

        batch_size: batch size dimension of this cell (Default: 1)
    """

    # Define Functions
    def __init__(
            self, name: str, n_units: int, target_freq: float = 63.75,
            batch_size: int = 1, key: Union[jax.Array, None] = None):
        super().__init__(name=name, key=key)

        _key, subkey = random.split(self.key.get(), 2)
        self.key.set(_key)

        ## Phasor meta-parameters
        self.target_freq = Compartment(target_freq, fixed=True)  ## maximum frequency (in Hertz/Hz)
        self.base_scale = Compartment(random.poisson(subkey[0], lam=target_freq, shape=(batch_size, n_units)) / target_freq, fixed=True)

        ## Layer Size Setup
        self.batch_size = Compartment(batch_size, fixed=True)
        self.n_units = Compartment(n_units, fixed=True)



        ## Compartment setup
        restVals = jnp.zeros((batch_size, n_units))
        self.inputs = Compartment(restVals,
                                  display_name="Input Stimulus")  # input
        # compartment
        self.outputs = Compartment(restVals,
                                   display_name="Spikes")  # output compartment
        self.tols = Compartment(initial_value=restVals,
                                display_name="Time-of-Last-Spike", units="ms")  # time of last spike
        self.angles = Compartment(restVals, display_name="Angles", units="deg")

    @compilable
    def advance_state(self, t, dt):
        ms_per_second = 1000  # ms/s
        events_per_ms = self.target_freq.get() / ms_per_second  # e/s s/ms -> e/ms
        ms_per_event = 1 / events_per_ms  # ms/e
        time_step_per_event = ms_per_event / dt  # ms/e * ts/ms -> ts / e
        angle_per_event = 2 * jnp.pi  # rad / e
        angle_per_timestep = angle_per_event / time_step_per_event  # rad / e
        # * e/ts -> rad / ts
        key, *subkey = random.split(self.key.get(), 3)

        scatter = ((random.normal(subkey[0], self.angles.get().shape) * 0.2) + 1) * self.base_scale.get()
        scattered_update = angle_per_timestep * scatter
        scaled_scattered_update = scattered_update * self.inputs.get()

        updated_angles = self.angles.get() + scaled_scattered_update
        self.outputs.set(jnp.where(updated_angles > angle_per_event, 1., 0.))

        self.angles.set(jnp.where(updated_angles > angle_per_event,
                                   updated_angles - angle_per_event,
                                   updated_angles))

        self.tols.set(self.tols.get() * (1. - self.outputs.get()) + t * self.outputs.get())

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size.get(), self.n_units.get()))
        not self.inputs.targeted and self.inputs.set(restVals)
        self.outputs.set(restVals)
        self.tols.set(restVals)
        self.angles.set(restVals)
        key, _ = random.split(self.key.get(), 2)
        self.key.set(key)


    @classmethod
    def help(cls):  ## component help function
        properties = {
            "cell_type": "Phasor - Produces input at a fairly regular "
                         "intervals with small amounts of noise)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"key": "JAX PRNG key",
                 "angles": "The current angle of the phasor"},
            "outputs":
                {"tols": "Time-of-last-spike",
                 "outputs": "Binary spike values emitted at time t"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "target_freq": "Maximum spike frequency of the (spike) train produced",
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "hyperparameters": hyperparams}
        return info


