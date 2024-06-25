from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.diffeq.ode_utils import step_euler
from ngclearn.utils.surrogate_fx import secant_lif_estimator
from ngclearn.utils import tensorstats

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

@partial(jit, static_argnums=[3,4])
def _modify_current(j, spikes, inh_weights, R_m, inh_R):
    """
    A simple function that modifies electrical current j via application of a
    scalar membrane resistance value and an approximate form of lateral inhibition.
    Note that if no inhibitory resistance is set (i.e., inh_R = 0), then no
    lateral inhibition is applied. Functionally, this routine carries out the
    following piecewise equation:

    | j * R_m - [Wi * s(t-dt)] * inh_R, if inh_R > 0
    | j * R_m, otherwise

    Args:
        j: electrical current value

        spikes: previous binary spike vector (for t-dt)

        inh_weights: lateral recurrent inhibitory synapses (typically should be
            chosen to be a scaled hollow matrix)

        R_m: membrane resistance (to multiply/scale j by)

        inh_R: inhibitory resistance to scale lateral inhibitory current by; if
            inh_R = 0, NO lateral inhibitory pressure will be applied

    Returns:
        modified electrical current value
    """
    _j = j * R_m
    if inh_R > 0.:
        _j = _j - (jnp.matmul(spikes, inh_weights) * inh_R)
    return _j

@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T): ## raw voltage dynamics
    mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    #dv_dt = ((-v + j) * (dt / tau_m)) * mask
    dv_dt = (-v + j)
    dv_dt = dv_dt * (1./tau_m) * mask
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T)
    return dv_dt

@jit
def _hyperpolarize(v, s):
    _v = (1. - s) * v ## hyper-polarize cells
    return _v

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

def _run_cell(dt, j, v, v_thr, tau_m, rfr, spike_fx, refract_T=1., thrGain=0.002,
              thrLeak=0.0005, rho_b = 0., sticky_spikes=False, v_min=None):
    """
    Runs leaky integrator neuronal dynamics

    Args:
        dt: integration time constant (milliseconds, or ms)

        j: electrical current value

        v: membrane potential (voltage) value (at t)

        v_thr: voltage threshold value (at t)

        tau_m: cell membrane time constant

        rfr: refractory variable vector (one per neuronal cell)

        spike_fx: spike emission function of form `spike_fx(v, v_thr)`

        refract_T: (relative) refractory time period (in ms; Default
            value is 1 ms)

        thrGain: the amount of threshold incremented per time step (if spike present)

        thrLeak: the amount of threshold value leaked per time step

        rho_b: sparsity factor; if > 0, will force adaptive threshold to operate
            with sparsity across a layer enforced

        sticky_spikes: if True, then spikes are pinned at value of action potential
            (i.e., 1) for as long as the relative refractory occurs (this recovers
            the source paper's core spiking process)

    Returns:
        voltage(t+dt), spikes, threshold(t+dt), updated refactory variables
    """
    #new_voltage, mask = _update_voltage(dt, j, v, rfr, tau_m, refract_T, v_min)
    v_params = (j, rfr, tau_m, refract_T)
    _, _v = step_euler(0., v, _dfv, dt, v_params) #_v = step_euler(v, v_params, _dfv, dt)
    # if v_min is not None:
    #     _v = jnp.maximum(v_min, _v)
    spikes = spike_fx(_v, v_thr)
    _v = _hyperpolarize(_v, spikes)
    new_thr = _update_threshold(dt, v_thr, spikes, thrGain, thrLeak, rho_b)
    _rfr, spikes = _update_refract_and_spikes(dt, rfr, spikes, refract_T, sticky_spikes)
    return _v, spikes, new_thr, _rfr

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

        rho_b: threshold sparsity factor (Default: 0)

        sticky_spikes: if True, spike variables will be pinned to action potential
            value (i.e, 1) throughout duration of the refractory period; this recovers
            a key setting used by Samadi et al., 2017

        thr_jitter: scale of uniform jitter to add to initialization of thresholds
    """

    # Define Functions
    def __init__(self, name, n_units, tau_m, resist_m, thr, resist_inh=0.,
                 thr_persist=False, thr_gain=0.0, thr_leak=0.0, rho_b=0.,
                 refract_time=0., sticky_spikes=False, thr_jitter=0.05, **kwargs):
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
        key, subkey = random.split(self.key.value)
        self.inh_weights = random.uniform(subkey, (n_units, n_units), minval=0.025, maxval=1.)
        self.inh_weights = self.inh_weights * (1. - jnp.eye(n_units))

        ## Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

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

    @staticmethod
    def _advance_state(t, dt, inh_weights, R_m, inh_R, d_spike_fx, tau_m,
                       spike_fx, refract_T, thrGain, thrLeak, rho_b,
                       sticky_spikes, v_min, j, s, v, thr, rfr, tols):
        ## run one step of Euler integration over neuronal dynamics
        j_curr = j
        ## apply simplified inhibitory pressure
        j_curr = _modify_current(j_curr, s, inh_weights, R_m, inh_R)
        j = j_curr # None ## store electrical current
        surrogate = d_spike_fx(j_curr, c1=0.82, c2=0.08)
        v, s, thr, rfr = \
            _run_cell(dt, j_curr, v, thr, tau_m,
                      rfr, spike_fx, refract_T, thrGain, thrLeak,
                      rho_b, sticky_spikes=sticky_spikes, v_min=v_min)
        ## update tols
        tols = _update_times(t, s, tols)
        return j, s, tols, v, thr, rfr, surrogate

    @resolver(_advance_state)
    def advance_state(self, j, s, tols, v, thr, rfr, surrogate):
        self.j.set(j)
        self.s.set(s)
        self.tols.set(tols)
        self.thr.set(thr)
        self.rfr.set(rfr)
        self.surrogate.set(surrogate)
        self.v.set(v)

    @staticmethod
    def _reset(refract_T, thr_persist, threshold0, batch_size, n_units, thr):
        restVals = jnp.zeros((batch_size, n_units))
        voltage = restVals
        refract = restVals + refract_T
        current = restVals
        surrogate = restVals + 1.
        timeOfLastSpike = restVals
        spikes = restVals
        if not thr_persist: ## if thresh non-persistent, reset to base value
            thr = threshold0 + 0
        return current, spikes, timeOfLastSpike, voltage, thr, refract, surrogate

    @resolver(_reset)
    def reset(self, j, s, tols, v, thr, rfr, surrogate):
        self.j.set(j)
        self.s.set(s)
        self.tols.set(tols)
        self.thr.set(thr)
        self.rfr.set(rfr)
        self.surrogate.set(surrogate)
        self.v.set(v)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.thr_persist == False:
            jnp.savez(file_name, threshold=self.threshold0) # save threshold0
        else:
            jnp.savez(file_name, threshold=self.thr.value) # save the actual threshold param/compartment

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.thr.set(data['threshold'])
        self.threshold0 = self.thr.value + 0

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
        X = SLIFCell("X", 9, 0.0004, 3, 0.3)
    print(X)
