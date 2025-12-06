# %%

from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn.utils.diffeq.ode_utils import step_euler
from ngclearn.utils.surrogate_fx import secant_lif_estimator

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T): ## raw voltage dynamics
    mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    #dv_dt = ((-v + j) * (dt / tau_m)) * mask
    dv_dt = (-v + j)
    dv_dt = dv_dt * (1./tau_m) * mask
    return dv_dt

#@partial(jit, static_argnums=[2])
def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T)
    return dv_dt

@partial(jit, static_argnums=[3,4,5])
def _update_threshold(dt, v_thr, spikes, thrGain=0.002, thrLeak=0.0005, rho_b = 0.):
    ## update thresholds if applicable
    if rho_b > 0.: ## run sparsity-enforcement threshold
        dthr = jnp.sum(spikes, axis=1, keepdims=True) - 1.0
        _v_thr = jnp.maximum(v_thr + dthr * rho_b, 0.025)
    else: ## run simple adaptive threshold
        thr_gain = spikes * thrGain
        thr_leak = (v_thr * thrLeak)
        _v_thr = v_thr + thr_gain - thr_leak
    return _v_thr

@partial(jit, static_argnums=[4])
def _update_refract_and_spikes(dt, rfr, s, refract_T, sticky_spikes=False):
    mask = (rfr >= refract_T).astype(jnp.float32) ## Note: wasted repeated compute
    ## update refractory variables
    _rfr = (rfr + dt) * (1. - s) + s * dt # set refract to dt
    _s = s
    if sticky_spikes == True: ## pin refractory spikes if configured
        _s = s * mask + (1. - mask)
    return _rfr, _s

class SLIFCell(JaxComponent): ## leaky integrate-and-fire cell
    """
    A spiking cell based on a simplified leaky integrate-and-fire (sLIF) model.
    This neuronal cell notably contains functionality required by the computational
    model employed by (Samadi et al., 2017, i.e., a surrogate derivative function
    and "sticky spikes") as well as the additional incorporation of an adaptive
    threshold (per unit) scheme. (Note that this particular spiking cell only
    supports Euler integration of its voltage dynamics.)

    | --- Cell Input Compartments: ---
    | j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | rfr - (relative) refractory variable state
    | thr - (adaptive) threshold state
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | surrogate - state of surrogate function output signals (currently, the secant LIF estimator)
    | tols - time-of-last-spike

    | Reference:
    | Samadi, Arash, Timothy P. Lillicrap, and Douglas B. Tweed. "Deep learning with
    | dynamic spiking neurons and fixed feedback weights." Neural computation 29.3
    | (2017): 578-602.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        resist_m: membrane resistance value

        thr: base value for adaptive thresholds (initial condition for
            per-cell thresholds) that govern short-term plasticity

        resist_inh: lateral modulation factor (DEFAULT: 6.); if >0, this will trigger
            a heuristic form of lateral inhibition via an internally integrated
            hollow matrix multiplication

        thr_persist: are adaptive thresholds persistent? (Default: False)

            :Note: depending on the value of this boolean variable:
                True = adaptive thresholds are NEVER reset upon call to reset
                False = adaptive thresholds are reset to "thr" upon call to reset

        thr_gain: how much adaptive thresholds increment by

        thr_leak: how much adaptive thresholds are decremented/decayed by

        refract_time: relative refractory period time (ms; Default: 1 ms)

        rho_b: threshold sparsity factor (Default: 0); note that setting rho_b > 0 will 
            force the adaptive threshold to follow dynamics that ignore `thr_grain` and 
            `thr_leak`

        sticky_spikes: if True, spike variables will be pinned to action potential
            value (i.e, 1) throughout duration of the refractory period; this recovers
            a key setting used by Samadi et al., 2017

        thr_jitter: scale of uniform jitter to add to initialization of thresholds

        batch_size: batch size dimension of this cell (Default: 1)
    """

    def __init__(
            self, name, n_units, tau_m, resist_m, thr, resist_inh=0., thr_persist=False, thr_gain=0.0, thr_leak=0.0,
            rho_b=0., refract_time=0., sticky_spikes=False, thr_jitter=0.05, batch_size=1, **kwargs
    ):
        super().__init__(name, **kwargs)

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.R_m = resist_m ## resistance value
        self.refract_T = refract_time #5. # 2. ## refractory period  # ms
        self.v_min = -3.
        ## variable below determines if spikes pinned at 1 during refractory period?
        self.sticky_spikes = sticky_spikes

        ## set up surrogate function for spike emission
        self.spike_fx, self.d_spike_fx = secant_lif_estimator()

        ## create simple recurrent inhibitory pressure
        self.inh_R = resist_inh ## lateral inhibitory magnitude
        key, subkey = random.split(self.key.get())
        self.inh_weights = random.uniform(subkey, (n_units, n_units), minval=0.025, maxval=1.)
        self.inh_weights = self.inh_weights * (1. - jnp.eye(n_units))

        ## Layer Size Setup
        self.n_units = n_units
        self.batch_size = batch_size

        ## Adaptive threshold setup
        self.rho_b = rho_b
        self.thr_persist = thr_persist ## are adapted thresholds persistent? True (persistent)
        self.thrGain = thr_gain #0.0005
        self.thrLeak = thr_leak #0.00005

        # thr_jitter: some random jitter to ensure thresholds start off different
        key, subkey = random.split(key)
        self.threshold0 = thr + random.uniform(subkey, (1, n_units),
                                               minval=-thr_jitter, maxval=thr_jitter,
                                               dtype=jnp.float32)

        ## Compartments
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals) ## electrical current, input
        self.s = Compartment(restVals) ## spike/action potential, output
        self.tols = Compartment(restVals) ## time-of-last-spike (record vector)
        self.v = Compartment(restVals) ## membrane potential/voltage
        self.thr = Compartment(self.threshold0 + 0.) ## action potential threshold
        self.rfr = Compartment(restVals + self.refract_T) ## refractory variable(s)
        self.surrogate = Compartment(restVals + 1.) ## surrogate signal

    @compilable
    def advance_state(self, t, dt):
        #####################################################################################
        #The following 3 lines of code modify electrical current j via application of a
        #scalar membrane resistance value and an approximate form of lateral inhibition.
        #Functionally, this routine carries out the following piecewise equation:
        #| j * R_m - [Wi * s(t-dt)] * inh_R, if inh_R > 0
        #| j * R_m, otherwise
        #| where j: electrical current value, spikes: previous binary spike vector (for t-dt), 
        #    inh_weights: lateral recurrent inhibitory synapses (typically should be chosen 
        #    to be a scaled hollow matrix), 
        #| R_m: membrane resistance (to multiply/scale j by), 
        #| inh_R: inhibitory resistance to scale lateral inhibitory current by; if inh_R = 0, 
        #    NO lateral inhibitory pressure will be applied

        # First, get the relevant compartment values
        j = self.j.get()
        # s = self.s.get() # NOTE: This is unused
        tols = self.tols.get()
        v = self.v.get()
        thr = self.thr.get()
        rfr = self.rfr.get()
        surrogate = self.surrogate.get()
        ## modify electrical current j via membrane resistance and lateral inhibition

        j = j * self.R_m
        if self.inh_R > 0.: ## if inh_R > 0, then lateral inhibition is applied
            j = j - (jnp.matmul(self.s.get(), self.inh_weights) * self.inh_R)
        #####################################################################################
        surrogate = self.d_spike_fx(j, c1=0.82, c2=0.08) ## calc surrogate deriv of spikes

        ## transition to:  voltage(t+dt), spikes, threshold(t+dt), refractory_variables(t+dt)
        v_params = (j, rfr, self.tau_m, self.refract_T)
        _, _v = step_euler(0., v, _dfv, dt, v_params)
        spikes = self.spike_fx(_v, thr)
        #_v = _hyperpolarize(_v, spikes)
        _v = (1. - spikes) * _v ## hyper-polarize cells
        new_thr = _update_threshold(dt, thr, spikes, self.thrGain, self.thrLeak, self.rho_b)
        _rfr, spikes = _update_refract_and_spikes(dt, rfr, spikes, self.refract_T, self.sticky_spikes)
        v = _v
        s = spikes
        thr = new_thr
        rfr = _rfr

        ## update tols
        tols = (1. - s) * tols + (s * t)
        # return j, s, tols, v, thr, rfr, surrogate
        self.j.set(j)
        self.s.set(s)
        self.tols.set(tols)
        self.v.set(v)
        self.thr.set(thr)
        self.rfr.set(rfr)
        self.surrogate.set(surrogate)

    @compilable
    def reset(self):
        # refract_T, thr_persist, threshold0, batch_size, n_units, thr
        restVals = jnp.zeros((self.batch_size, self.n_units))
        voltage = restVals
        refract = restVals + self.refract_T
        current = restVals
        surrogate = restVals + 1.
        timeOfLastSpike = restVals
        spikes = restVals
        if not self.thr_persist: ## if thresh non-persistent, reset to base value
            thr = self.threshold0 + 0
            self.thr.set(thr)
        # return current, spikes, timeOfLastSpike, voltage, thr, refract, surrogate
        self.j.set(current)
        self.s.set(spikes)
        self.tols.set(timeOfLastSpike)
        self.v.set(voltage)
        self.rfr.set(refract)
        self.surrogate.set(surrogate)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.thr_persist == False:
            jnp.savez(file_name, threshold=self.threshold0) # save threshold0
        else:
            jnp.savez(file_name, threshold=self.thr.get()) # save the actual threshold param/compartment

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.thr.set(data['threshold'])
        self.threshold0 = self.thr.get() + 0

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "SLIFCell - evolves neurons according to simplified "
                         "leaky integrate-and-fire spiking dynamics."
        }
        compartment_props = {
            "inputs":
                {"j": "External input electrical current"},
            "states":
                {"v": "Membrane potential/voltage at time t",
                 "rfr": "Current state of (relative) refractory variable",
                 "thr": "Current state of voltage threshold at time t",
                 "key": "JAX PRNG key"},
            "outputs":
                {"s": "Emitted spikes/pulses at time t",
                 "tols": "Time-of-last-spike",
                 "surrogate": "State/value of surrogate function at time t"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "tau_m": "Cell membrane time constant",
            "resist_m": "Membrane resistance value",
            "thr": "Base voltage threshold value",
            "resist_inh": "Inhibitory resistance value",
            "thr_persist": "Should adaptive threshold persist across reset calls?",
            "thr_gain": "Amount to increment threshold by upon occurrence of spike",
            "thr_leak": "Amount to decay threshold upon occurrence of spike",
            "rho_b": "Shared threshold sparsity control parameter (if using shared threshold)",
            "refract_time": "Length of relative refractory period (ms)",
            "thr_jitter": "Scale of random uniform noise to apply to initial condition of threshold",
            "sticky_spikes": "Should spikes be allowed to persist during refractory period?"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = -v + j * resist_m",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = SLIFCell("X", 9, 0.0004, 3, 0.3)
    print(X)
