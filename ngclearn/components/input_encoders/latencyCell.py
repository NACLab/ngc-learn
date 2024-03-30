from ngcsimlib.component import Component
from jax import numpy as jnp, random, jit
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

@jit
def calc_spike_times_linear(data_in, tau, thr):
    """
    Computes spike times from data according to a

    Args:
        spk_times: spike times to compare against

        t: current time

    Returns:
        binary spikes
    """
    #torch.clamp_max((-tau * (data - 1)), -tau * (threshold - 1))
    stimes = jnp.fmax(-tau * (data_in - 1.), -tau * (thr - 1.))
    return stimes

@jit
def calc_spike_times_nonlinear(data_in, tau, thr):
    """
    Extracts a spike from a latency-coding spike train.

    Args:
        spk_times: spike times to compare against

        t: current time

    Returns:
        binary spikes
    """
    stimes = jnp.log(data_in / (data_in - thr)) * tau ## calc spike times
    return stimes

@jit
def extract_spike(spk_times, t):
    """
    Extracts a spike from a latency-coded spike train.

    Args:
        spk_times: spike times to compare against

        t: current time

    Returns:
        binary spikes
    """
    spikes_t = (spk_times <= t).astype(jnp.float32) # get spike
    return spikes_t

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
    def __init__(self, name, n_units, tau=1., threshold=0.01, key=None,
                 useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        self.tau = tau
        self.threshold = threshold
        self.linearize = False
        ## normalize latency code s.t. final spike(s) occur w/in num_steps
        # self.normalize = False
        # self.num_steps = 100.

        self.target_spike_times = None

        ##Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units
        self.reset()

    def verify_connections(self):
        pass

    def advance_state(self, t, dt, **kwargs):
        self.key, *subkeys = random.split(self.key, 2)
        tau = self.tau
        # if self.normalize == True:
        #     tau = num_steps - 1. - first_spike_time ## linear normalization
        data = self.inputCompartment ## get sensory pattern data / features
        if self.target_spike_times == None: ## calc spike times
            if self.linearize == True: ## linearize spike time calculation
                stimes = calc_spike_times_linear(data, self.tau, self.threshold)
                self.target_spike_times = stimes
            else: ## standard nonlinear spike time calculation
                stimes = calc_spike_times_nonlinear(data, self.tau, self.threshold)
                self.target_spike_times = stimes
        spikes = extract_spike(self.target_spike_times, t) ## get spikes at t
        self.outputCompartment = spikes
        self.timeOfLastSpike = update_times(t, self.outputCompartment, self.timeOfLastSpike)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.outputCompartment = jnp.zeros((self.batch_size, self.n_units)) #None
        self.timeOfLastSpike = jnp.zeros((self.batch_size, self.n_units))
        self.target_spike_times = None

    def save(self, **kwargs):
        pass
