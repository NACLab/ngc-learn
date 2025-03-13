from ngclearn import resolver, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from jax import numpy as jnp, random
from ngcsimlib.logger import warn

class PhasorCell(JaxComponent):
    """
    A phasor cell that emits a pulse at a regular interval.

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

        target_freq: maximum frequency (in Hertz) of this spike train
            (must be > 0.)
    """

    # Define Functions
    def __init__(self, name, n_units, target_freq=63.75, batch_size=1,
                 **kwargs):
        super().__init__(name, **kwargs)

        ## Phasor meta-parameters
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
        self.tols = Compartment(initial_value=restVals,
                                display_name="Time-of-Last-Spike", units="ms")  # time of last spike
        self.angles = Compartment(restVals, display_name="Angles", units="deg")
        # self.base_scale = random.uniform(subkey, self.angles.value.shape,
        #                                  minval=0.75, maxval=1.25)
        # self.base_scale = ((random.normal(subkey, self.angles.value.shape) * 0.15) + 1)
        # alpha = ((random.normal(subkey, self.angles.value.shape) * (jnp.sqrt(target_freq) / target_freq)) + 1)
        # beta = random.poisson(subkey, lam=target_freq, shape=self.angles.value.shape) / target_freq

        self.base_scale = random.poisson(subkey, lam=target_freq, shape=self.angles.value.shape) / target_freq

    def validate(self, dt=None, **validation_kwargs):
        valid = super().validate(**validation_kwargs)
        if dt is None:
            warn(f"{self.name} requires a validation kwarg of `dt`")
            return False
        ## check for unstable combinations of dt and target-frequency
        # meta-params
        events_per_timestep = (
                                  dt / 1000.) * self.target_freq  ##
        # compute scaled probability
        if events_per_timestep > 1.:
            valid = False
            warn(
                f"{self.name} will be unable to make as many temporal events "
                f"as "
                f"requested! ({events_per_timestep} events/timestep) Unstable "
                f"combination of dt = {dt} and target_freq = "
                f"{self.target_freq} "
                f"being used!"
            )
        return valid

    @staticmethod
    def _advance_state(t, dt, target_freq, key,
                       inputs, angles, tols, base_scale):
        ms_per_second = 1000  # ms/s
        events_per_ms = target_freq / ms_per_second  # e/s s/ms -> e/ms
        ms_per_event = 1 / events_per_ms  # ms/e
        time_step_per_event = ms_per_event / dt  # ms/e * ts/ms -> ts / e
        angle_per_event = 2 * jnp.pi  # rad / e
        angle_per_timestep = angle_per_event / time_step_per_event  # rad / e
        # * e/ts -> rad / ts
        key, subkey = random.split(key, 2)
        # scatter = random.uniform(subkey, angles.shape, minval=0.5,
        #                          maxval=1.5) * base_scale

        scatter = ((random.normal(subkey, angles.shape) * 0.2) + 1) * base_scale
        scattered_update = angle_per_timestep * scatter
        scaled_scattered_update = scattered_update * inputs

        updated_angles = angles + scaled_scattered_update
        outputs = jnp.where(updated_angles > angle_per_event, 1., 0.)
        updated_angles = jnp.where(updated_angles > angle_per_event,
                                   updated_angles - angle_per_event,
                                   updated_angles)
        tols = tols * (1. - outputs) + t * outputs

        return outputs, tols, key, updated_angles

    @resolver(_advance_state)
    def advance_state(self, outputs, tols, key, angles):
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.key.set(key)
        self.angles.set(angles)

    @staticmethod
    def _reset(batch_size, n_units, key, target_freq):
        restVals = jnp.zeros((batch_size, n_units))
        key, subkey = random.split(key, 2)
        return restVals, restVals, restVals, restVals, key

    @resolver(_reset)
    def reset(self, inputs, outputs, tols, angles, key):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.key.set(key)
        self.angles.set(angles)

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
            "target_freq": "Maximum spike frequency of the train produced",
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
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


