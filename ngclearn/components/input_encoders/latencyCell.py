from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from functools import partial
import jax
from typing import Union

from ngclearn.utils.model_utils import clamp_min, clamp_max

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

@partial(jit, static_argnums=[5])
def _calc_spike_times_linear(data, tau, thr, first_spk_t, num_steps=1.,
                             normalize=False):
    """
    Computes spike times from data according to a linear latency encoding scheme.

    Args:
        data: pattern data to convert to spikes/times

        tau: latency coding time constant

        thr: latency coding threshold value

        first_spk_t: first spike time(s) (either int or vector
            with same shape as spk_times; in ms)

        num_steps: number of total time steps of simulation to consider

        normalize: normalize the logarithmic latency code values (uses num_steps)

    Returns:
        projected spike times
    """
    _tau = tau
    if normalize:
        _tau = num_steps - 1. - first_spk_t ## linear normalization
    #torch.clamp_max((-tau * (data - 1)), -tau * (threshold - 1))
    stimes = -_tau * (data - 1.) ## calc raw latency code values
    max_bound = -_tau * (thr - 1.) ## upper bound latency code values
    stimes = clamp_max(stimes, max_bound) ## apply upper bound
    return stimes + first_spk_t

@partial(jit, static_argnums=[6])
def _calc_spike_times_nonlinear(data, tau, thr, first_spk_t, eps=1e-7,
                                num_steps=1., normalize=False):
    """
    Computes spike times from data according to a logarithmic encoding scheme.

    Args:
        data: pattern data to convert to spikes/times

        tau: latency coding time constant

        thr: latency coding threshold value

        first_spk_t: first spike time(s) (either int or vector
            with same shape as spk_times; in ms)

        eps: small numerical error control factor (added to thr)

        num_steps: number of total time steps of simulation to consider

        normalize: normalize the logarithmic latency code values (uses num_steps)

    Returns:
        projected spike times
    """
    _data = clamp_min(data, thr + eps) # saturates all values below threshold.
    stimes = jnp.log(_data / (_data - thr)) * tau ## calc spike times
    stimes = stimes + first_spk_t

    if normalize:
        term1 = (stimes - first_spk_t)
        term2 = (num_steps - first_spk_t - 1.)
        term3 = jnp.max(stimes - first_spk_t)
        stimes = term1 * (term2 / term3) + first_spk_t
    return stimes

@jit
def _extract_spike(spk_times, t, mask):
    """
    Extracts a spike from a latency-coded spike train.

    Args:
        spk_times: spike times to compare against

        t: current time

        mask: prior spike mask (1 if spike has occurred, 0 otherwise)

    Returns:
        binary spikes, boolean mask to indicate if spikes have occurred as of yet
    """
    _spk_times = jnp.round(spk_times) # snap times to nearest integer time
    spikes_t = (_spk_times <= t).astype(jnp.float32) # get spike
    spikes_t = spikes_t * (1. - mask)
    _mask = mask + (1. - mask) * spikes_t
    return spikes_t, _mask

class LatencyCell(JaxComponent):
    """
    A (nonlinear) latency encoding (spike) cell; produces a time-lagged set of
    spikes on-the-fly.

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell State Compartments: ---
    | targ_sp_times - target-spike-time
    | mask - spike-ordering mask
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | outputs - output
    | tols - time-of-last-spike

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau: time constant for model used to calculate firing time (Default: 1 ms)

        threshold: sensory input features below this threhold value will fire at
            final step in time of this latency coded spike train

        first_spike_time: time of first allowable spike (ms) (Default: 0 ms)

        linearize: should the linear latency encoding scheme be used? (otherwise,
            defaults to logarithmic latency encoding)

        normalize: normalize the latency code such that final spike(s) occur
            a pre-specified number of simulation steps "num_steps"? (Default: False)

            :Note: if this set to True, you will need to choose a useful value
                for the "num_steps" argument (>1), depending on how many steps simulated

        clip_spikes: should values under threshold be removed/suppressed?
            (default: False)

        num_steps: number of discrete time steps to consider for normalized latency
            code (only useful if "normalize" is set to True) (Default: 1)

        batch_size: batch size dimension of this cell (Default: 1)
    """

    def __init__(
            self, name: str, n_units: int, tau: float = 1., threshold: float = 0.01, first_spike_time: float = 0.,
            linearize: bool = False, normalize: bool = False, clip_spikes: bool = False, num_steps: float = 1.,
            batch_size: int = 1, key: Union[jax.Array, None] = None, **kwargs
    ):
        super().__init__(name=name, key=key)

        ## latency meta-parameters
        self.first_spike_time = Compartment(first_spike_time)
        self.tau = Compartment(tau)
        self.threshold = Compartment(threshold)
        self.linearize = Compartment(linearize)
        self.clip_spikes = Compartment(clip_spikes)
        ## normalize latency code s.t. final spike(s) occur w/in num_steps
        self.normalize = Compartment(normalize)
        self.num_steps = Compartment(num_steps)

        ## Layer Size Setup
        self.batch_size = Compartment(batch_size)
        self.n_units = Compartment(n_units)

        ## Compartment setup
        restVals = jnp.zeros((batch_size, n_units))
        self.inputs = Compartment(restVals, display_name="Input Stimulus") # input compartment
        self.outputs = Compartment(restVals, display_name="Spikes") # output compartment
        self.mask = Compartment(restVals, display_name="Spike Time Mask")
        self.clip_mask = Compartment(restVals, display_name="Clip Mask")
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike", units="ms") # time of last spike
        self.targ_sp_times = Compartment(restVals, display_name="Target Spike Time", units="ms")

    @compilable
    def calc_spike_times(self):
        if self.clip_spikes.get():
            self.clip_mask.set((self.inputs.get() < self.threshold) * 1.)
        else:
            self.clip_mask.set(self.inputs.get() * 0.)

        if self.linearize.get():
            self.targ_sp_times.set(
                _calc_spike_times_linear(self.inputs.get(),
                                         self.tau.get(),
                                         self.threshold.get(),
                                         self.first_spike_time.get(),
                                         self.num_steps.get(),
                                         self.normalize.get()))
        else:
            self.targ_sp_times.set(
                _calc_spike_times_nonlinear(self.inputs.get(),
                                            self.tau.get(),
                                            self.threshold.get(),
                                            self.first_spike_time.get(),
                                            self.num_steps.get(),
                                            self.normalize.get()))


    @compilable
    def advance_state(self, t):
        spikes, spike_mask = _extract_spike(self.targ_sp_times.get(), t, self.mask.get())
        self.tols.set((1. - spikes) * self.tols.get() + (spikes * t))
        self.outputs.set(spikes * (1. - self.clip_mask.get()))
        self.mask.set(spike_mask)

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size.get(), self.n_units.get()))
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(restVals)
        self.outputs.set(restVals)
        self.tols.set(restVals)
        self.mask.set(restVals)
        self.clip_mask.set(restVals)
        self.targ_sp_times.set(restVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "LatencyCell - samples input to produce spikes via latency "
                         "coding, where each dimension's magnitude determines how "
                         "early in the spike train a value occurs. This is a "
                         "temporal/order encoder."
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"targ_sp_times": "Target spike times",
                 "mask": "Spike ordering mask",
                 "key": "JAX PRNG key"},
            "outputs":
                {"tols": "Time-of-last-spike",
                 "outputs": "Binary spike values emitted at time t"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "threshold": "Spike threshold (constant and shared across neurons)",
            "linearize": "Should a linear latency encoding be used?",
            "normalize": "Should the latency code(s) be normalized?",
            "num_steps": "Number of total time steps of simulation to consider ("
                         "useful for target spike time computation",
            "first_spike_time": "Time of first allowable spike"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "~ Latency(x)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = LatencyCell("X", 9)
    print(X)
    print(X.calc_spike_times.compiled.code)
    print(X.advance_state.compiled.code)
