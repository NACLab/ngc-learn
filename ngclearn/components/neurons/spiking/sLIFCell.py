from ngclib.component import Component
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

@jit
def update_times(t, s, tols):
    _tols = (1. - s) * tols + (s * t)
    return _tols

@jit
def apply_surrogate_fx(j, c1=0.82, c2=0.08):
    # sech(x) = 1/cosh(x), cosh(x) = (e^x + e^(-x))/2
    # c1 c2 sech^2(c2 j) for j > 0 and 0 for j <= 0
    mask = (j > 0.).astype(jnp.float32)
    dj = j * c2
    cosh_j = (jnp.exp(dj) + jnp.exp(-dj))/2.
    sech_j = 1./cosh_j #(cosh_x + 1e-6)
    dv_dj = sech_j * (c1 * c2) # ~deriv w.r.t. j
    return dv_dj * mask ## 0 if j < 0, otherwise, use dv/dj for j >= 0

@partial(jit, static_argnums=[3,4])
def modify_current(j, spikes, inh_weights, R_m, inh_R):
    _j = j * R_m
    if inh_R > 0.:
        _j = _j - (jnp.matmul(spikes, inh_weights) * inh_R)
    return _j

@partial(jit, static_argnums=[6,7,8,9])
def run_cell(dt, j, v, v_thr, tau_m, rfr, refract_T=1., thrGain=0.002,
             thrLeak=0.0005, sticky_spikes=False):
    """
    Runs leaky integrator neuronal dynamics
    """
    mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    new_voltage = (v + (-v + j) * (dt / tau_m)) * mask
    #new_voltage = v + (-v) * (dt / tau_m) + j * mask
    spikes = jnp.where(new_voltage > v_thr, 1, 0)
    new_voltage = (1. - spikes) * new_voltage ## hyper-polarize cells
    ## update thresholds if applicable
    thr_gain = spikes * thrGain
    thr_leak = (v_thr * thrLeak)
    new_thr = v_thr + thr_gain - thr_leak
    ## update refractory variables
    _rfr = (rfr + dt) * (1. - spikes)
    if sticky_spikes == True: ## pin refractory spikes if configured
        spikes = spikes * mask + (1. - mask)
    return new_voltage, spikes, new_thr, _rfr

class SLIFCell(Component): ## leaky integrate-and-fire cell
    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'j' ## electrical current

    @classmethod
    def outputCompartmentName(cls):
        return 's' ## spike/action potential

    @classmethod
    def timeOfLastSpikeCompartmentName(cls):
        return 'tols' ## time-of-last-spike (record vector)

    @classmethod
    def voltageCompartmentName(cls):
        return 'v' ## membrane potential/voltage

    @classmethod
    def thresholdCompartmentName(cls):
        return 'thr' ## action potential threshold

    @classmethod
    def refractCompartmentName(cls):
        return 'rfr' ## refractory variable(s)

    @classmethod
    def surrogateCompartmentName(cls):
        return 'surrogate'

    ## Bind Properties to Compartments for ease of use
    @property
    def current(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @current.setter
    def current(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError(
                    "Input Compartment size does not match provided input size " + str(inp.shape) + "for "
                    + str(self.name))
        self.compartments[self.inputCompartmentName()] = inp

    @property
    def surrogate(self):
        return self.compartments.get(self.surrogateCompartmentName(), None)

    @surrogate.setter
    def surrogate(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError(
                    "Surrogate function Compartment size does not match provided input size " + str(inp.shape) + "for "
                    + str(self.name))
        self.compartments[self.surrogateCompartmentName()] = inp

    @property
    def spikes(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @spikes.setter
    def spikes(self, out):
        if out is not None:
            if out.shape[1] != self.n_units:
                raise RuntimeError(
                    "Output compartment size (n, " + str(self.n_units) + ") does not match provided output size "
                    + str(out.shape) + " for " + str(self.name))
        self.compartments[self.outputCompartmentName()] = out

    @property
    def timeOfLastSpike(self):
        return self.compartments.get(self.timeOfLastSpikeCompartmentName(), None)

    @timeOfLastSpike.setter
    def timeOfLastSpike(self, t):
        if t is not None:
            if t.shape[1] != self.n_units:
                raise RuntimeError("Time of last spike compartment size (n, " + str(self.n_units) +
                                   ") does not match provided size " + str(t.shape) + " for " + str(self.name))
        self.compartments[self.timeOfLastSpikeCompartmentName()] = t

    @property
    def voltage(self):
        return self.compartments.get(self.voltageCompartmentName(), None)

    @voltage.setter
    def voltage(self, v):
        if v is not None:
            if v.shape[1] != self.n_units:
                raise RuntimeError("Time of last spike compartment size (n, " + str(self.n_units) +
                                   ") does not match provided size " + str(v.shape) + " for " + str(self.name))
        self.compartments[self.voltageCompartmentName()] = v

    @property
    def refract(self):
        return self.compartments.get(self.refractCompartmentName(), None)

    @refract.setter
    def refract(self, rfr):
        if rfr is not None:
            if rfr.shape[1] != self.n_units:
                raise RuntimeError("Refractory variable compartment size (n, " + str(self.n_units) +
                                   ") does not match provided size " + str(rfr.shape) + " for " + str(self.name))
        self.compartments[self.refractCompartmentName()] = rfr

    @property
    def threshold(self):
        return self.compartments.get(self.thresholdCompartmentName(), None)

    @threshold.setter
    def threshold(self, thr):
        self.compartments[self.thresholdCompartmentName()] = thr

    # Define Functions
    def __init__(self, name, n_units, tau_m, R_m, thr, inhibit_R=6., thr_persist=False,
                 thrGain=0.001, thrLeak=0.00005, refract_T=1., key=None, useVerboseDict=False,
                 directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.R_m = R_m ## resistance value
        self.refract_T = refract_T #5. # 2. ## refractory period  # ms

        ## create simple recurrent inhibitory pressure
        self.inh_R = inhibit_R ## lateral inhibitory magnitude
        self.key, subkey = random.split(self.key)
        self.inh_weights = random.uniform(subkey, (n_units, n_units), minval=0.025, maxval=1.)
        MV = 1. - jnp.eye(n_units)
        self.inh_weights = self.inh_weights * MV

        ##Layer Size Setup
        self.n_units = n_units

        ## adaptive threshold setup
        self.thr_persist = thr_persist ## are adapted thresholds persistent? True (persistent)
        self.thrGain = thrGain #0.0005
        self.thrLeak = thrLeak #0.00005
        if directory is None:
            self.thr_jitter = 0.05 ## some random jitter to ensure thresholds start off different
            self.key, subkey = random.split(self.key)
            #self.threshold = random.uniform(subkey, (1, n_units), minval=thr, maxval=1.25 * thr)
            self.threshold0 = thr + random.uniform(subkey, (1, n_units),
                                                  minval=-self.thr_jitter, maxval=self.thr_jitter,
                                                  dtype=jnp.float32)
            self.threshold = self.threshold0 + 0 ## save initial threshold
        else:
            self.load(directory)

        ## Set up bundle for multiple inputs of current
        self.create_bundle('multi_input', 'additive')
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        if self.spikes is None:
            self.spikes = jnp.zeros((1, self.n_units))
        if self.refract is None:
            self.refract = jnp.zeros((1, self.n_units)) + self.refract_T
        ## run one step of Euler integration over neuronal dynamics
        j_curr = self.current
        ## apply simplified inhibitory pressure
        j_curr = modify_current(j_curr, self.spikes, self.inh_weights, self.R_m, self.inh_R)
        self.current = j_curr # None ## store electrical current
        self.surrogate = apply_surrogate_fx(j_curr, c1=0.82, c2=0.08)
        self.voltage, self.spikes, self.threshold, self.refract = \
            run_cell(dt, j_curr, self.voltage, self.threshold, self.tau_m,
                     self.refract, self.refract_T, self.thrGain, self.thrLeak,
                     sticky_spikes=True)
        ## update tols
        self.timeOfLastSpike = update_times(t, self.spikes, self.timeOfLastSpike)
        #self.timeOfLastSpike = (1 - self.spikes) * self.timeOfLastSpike + (self.spikes * t)

    def reset(self, **kwargs):
        self.voltage = jnp.zeros((1, self.n_units))
        self.refract = jnp.zeros((1, self.n_units)) + self.refract_T
        self.current = None
        self.surrogate = None
        self.timeOfLastSpike = jnp.zeros((1, self.n_units))
        self.spikes = jnp.zeros((1, self.n_units)) #None
        if self.thr_persist == False: ## if thresh non-persistent, reset to base value
            self.threshold = self.threshold0 + 0

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.thr_persist == False:
            jnp.savez(file_name, threshold=self.threshold0)
        else:
            jnp.savez(file_name, threshold=self.threshold)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.threshold = data['threshold']
        self.threshold0 = self.threshold + 0
