from ngcsimlib.component import Component
from ngclearn.utils.model_utils import clamp_min, clamp_max
from jax import numpy as jnp, random, jit
from functools import partial
import time

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

@partial(jit, static_argnums=[5])
def calc_spike_times_linear(data, tau, thr, first_spk_t, num_steps=1.,
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
def calc_spike_times_nonlinear(data, tau, thr, first_spk_t, eps=1e-7,
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
def extract_spike(spk_times, t, mask):
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

class LatencyCell(Component):
    """
    A (nonlinear) latency encoding (spike) cell; produces a time-lagged set of
    spikes on-the-fly.

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

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'in'

    @classmethod
    def outputCompartmentName(cls):
        return 'out'

    @classmethod
    def timeOfLastSpikeCompartmentName(cls):
        return 'tols'

    ## Bind Properties to Compartments for ease of use
    @property
    def inputCompartment(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, inp):
        self.compartments[self.inputCompartmentName()] = inp

    @property
    def outputCompartment(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @outputCompartment.setter
    def outputCompartment(self, out):
        self.compartments[self.outputCompartmentName()] = out

    @property
    def timeOfLastSpike(self):
        return self.compartments.get(self.timeOfLastSpikeCompartmentName(), None)

    @timeOfLastSpike.setter
    def timeOfLastSpike(self, t):
        self.compartments[self.timeOfLastSpikeCompartmentName()] = t

    # Define Functions
    def __init__(self, name, n_units, tau=1., threshold=0.01, first_spike_time=0.,
                 linearize=False, normalize=False, num_steps=1., key=None,
                 useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        self.first_spike_time = first_spike_time
        self.tau = tau
        self.threshold = threshold
        self.linearize = linearize
        ## normalize latency code s.t. final spike(s) occur w/in num_steps
        self.normalize = normalize
        self.num_steps = num_steps

        self.target_spike_times = None
        self._mask = None

        ##Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units
        self.reset()

    def verify_connections(self):
        pass

    def advance_state(self, t, dt, **kwargs):
        self.key, *subkeys = random.split(self.key, 2)
        tau = self.tau
        data = self.inputCompartment ## get sensory pattern data / features

        if self.target_spike_times == None: ## calc spike times if not called yet
            if self.linearize == True: ## linearize spike time calculation
                stimes = calc_spike_times_linear(data, tau, self.threshold,
                                                 self.first_spike_time,
                                                 self.num_steps, self.normalize)
                self.target_spike_times = stimes
            else: ## standard nonlinear spike time calculation
                stimes = calc_spike_times_nonlinear(data, tau, self.threshold,
                                                    self.first_spike_time,
                                                    num_steps=self.num_steps,
                                                    normalize=self.normalize)
                self.target_spike_times = stimes
            #print(self.target_spike_times)
            #sys.exit(0)
        spk_mask = self._mask
        spikes, spk_mask = extract_spike(self.target_spike_times, t, spk_mask) ## get spikes at t
        self._mask = spk_mask
        self.outputCompartment = spikes
        self.timeOfLastSpike = update_times(t, self.outputCompartment, self.timeOfLastSpike)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = jnp.zeros((self.batch_size, self.n_units)) #None
        self.timeOfLastSpike = jnp.zeros((self.batch_size, self.n_units))
        self._mask = jnp.zeros((self.batch_size, self.n_units))

    def save(self, **kwargs):
        pass
