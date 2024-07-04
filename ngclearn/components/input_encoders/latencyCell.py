from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngclearn.utils.model_utils import clamp_min, clamp_max
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
    if normalize == True:
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

    if normalize == True:
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

        num_steps: number of discrete time steps to consider for normalized latency
            code (only useful if "normalize" is set to True) (Default: 1)
    """

    # Define Functions
    def __init__(self, name, n_units, tau=1., threshold=0.01, first_spike_time=0.,
                 linearize=False, normalize=False, num_steps=1.,
                 batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        ## latency meta-parameters
        self.first_spike_time = first_spike_time
        self.tau = tau
        self.threshold = threshold
        self.linearize = linearize
        ## normalize latency code s.t. final spike(s) occur w/in num_steps
        self.normalize = normalize
        self.num_steps = num_steps

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input compartment
        self.outputs = Compartment(restVals) # output compartment
        self.mask = Compartment(restVals)  # output compartment
        self.tols = Compartment(restVals) # time of last spike
        self.targ_sp_times = Compartment(restVals)
        #self.reset()

    @staticmethod
    def _calc_spike_times(linearize, tau, threshold, first_spike_time, num_steps,
                          normalize, inputs):
        ## would call this function before processing a spike train (at start)
        data = inputs
        if linearize == True: ## linearize spike time calculation
            stimes = _calc_spike_times_linear(data, tau, threshold,
                                              first_spike_time,
                                              num_steps, normalize)
            targ_sp_times = stimes #* calcEvent + targ_sp_times * (1. - calcEvent)
        else: ## standard nonlinear spike time calculation
            stimes = _calc_spike_times_nonlinear(data, tau, threshold,
                                                 first_spike_time,
                                                 num_steps=num_steps,
                                                 normalize=normalize)
            targ_sp_times = stimes #* calcEvent + targ_sp_times * (1. - calcEvent)
        return targ_sp_times

    @resolver(_calc_spike_times)
    def calc_spike_times(self, targ_sp_times):
        self.targ_sp_times.set(targ_sp_times)

    @staticmethod
    def _advance_state(t, dt, key, inputs, mask, targ_sp_times, tols):
        key, *subkeys = random.split(key, 2)
        data = inputs ## get sensory pattern data / features
        spikes, spk_mask = _extract_spike(targ_sp_times, t, mask) ## get spikes at t
        tols = _update_times(t, spikes, tols)
        return spikes, tols, spk_mask, targ_sp_times, key

    @resolver(_advance_state)
    def advance_state(self, outputs, tols, mask, targ_sp_times, key):
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.mask.set(mask)
        self.targ_sp_times.set(targ_sp_times)
        self.key.set(key)

    @staticmethod
    def _reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        return (restVals, restVals, restVals, restVals, restVals)

    @resolver(_reset)
    def reset(self, inputs, outputs, tols, mask, targ_sp_times):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.mask.set(mask)
        self.targ_sp_times.set(targ_sp_times)

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
        X = LatencyCell("X", 9)
    print(X)
