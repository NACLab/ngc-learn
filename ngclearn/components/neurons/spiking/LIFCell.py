from jax import numpy as jnp, random, jit, nn
from functools import partial
from ngclearn.utils import tensorstats
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2
from ngclearn.utils.surrogate_fx import secant_lif_estimator, arctan_estimator, triangular_estimator

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

# @jit
# def _modify_current(j, dt, tau_m, R_m):
#     ## electrical current re-scaling co-routine
#     jScale = tau_m/dt ## <-- this anti-scale counter-balances form of ODE used in this cell
#     return (j * R_m) * jScale

@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_decay=1.): ## raw voltage dynamics
    mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    ## update voltage / membrane potential
    dv_dt = (v_rest - v) * v_decay + (j * mask)
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T, v_rest, v_decay = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_decay)
    return dv_dt

#@partial(jit, static_argnums=[7, 8, 9, 10, 11, 12])
def _run_cell(dt, j, v, v_thr, v_theta, rfr, skey, tau_m, v_rest, v_reset,
              v_decay, refract_T, integType=0):
    """
    Runs leaky integrator (or leaky integrate-and-fire; LIF) neuronal dynamics.

    Args:
        dt: integration time constant (milliseconds, or ms)

        j: electrical current value

        v: membrane potential (voltage, in milliVolts or mV) value (at t)

        v_thr: base voltage threshold value (in mV)

        v_theta: threshold shift (homeostatic) variable (at t)

        rfr: refractory variable vector (one per neuronal cell)

        skey: PRNG key which, if not None, will trigger a single-spike constraint
            (i.e., only one spike permitted to emit per single step of time);
            specifically used to randomly sample one of the possible action
            potentials to be an emitted spike

        tau_m: cell membrane time constant

        v_rest: membrane resting potential (in mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value

        v_decay: strength of voltage leak (Default: 1.)

        refract_T: (relative) refractory time period (in ms; Default
            value is 1 ms)

        integType: integer indicating type of integration to use

    Returns:
        voltage(t+dt), spikes, raw spikes, updated refactory variables
    """
    _v_thr = v_theta + v_thr ## calc present voltage threshold
    #mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    ## update voltage / membrane potential
    v_params = (j, rfr, tau_m, refract_T, v_rest, v_decay)
    if integType == 1:
        _, _v = step_rk2(0., v, _dfv, dt, v_params)
    else: #_v = v + (v_rest - v) * (dt/tau_m) + (j * mask)
        _, _v = step_euler(0., v, _dfv, dt, v_params)
    ## obtain action potentials/spikes
    s = (_v > _v_thr).astype(jnp.float32)
    ## update refractory variables
    _rfr = (rfr + dt) * (1. - s)
    ## perform hyper-polarization of neuronal cells
    _v = _v * (1. - s) + s * v_reset

    raw_s = s + 0 ## preserve un-altered spikes
    ############################################################################
    ## this is a spike post-processing step
    if skey is not None:
        m_switch = (jnp.sum(s) > 0.).astype(jnp.float32) ## TODO: not batch-able
        rS = s * random.uniform(skey, s.shape)
        rS = nn.one_hot(jnp.argmax(rS, axis=1), num_classes=s.shape[1],
                        dtype=jnp.float32)
        s = s * (1. - m_switch) + rS * m_switch
    ############################################################################
    return _v, s, raw_s, _rfr

@partial(jit, static_argnums=[3, 4])
def _update_theta(dt, v_theta, s, tau_theta, theta_plus=0.05):
    """
    Runs homeostatic threshold update dynamics one step (via Euler integration).

    Args:
        dt: integration time constant (milliseconds, or ms)

        v_theta: current value of homeostatic threshold variable

        s: current spikes (at t)

        tau_theta: homeostatic threshold time constant

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

    Returns:
        updated homeostatic threshold variable
    """
    #theta_decay = 0.9999999 #0.999999762 #jnp.exp(-dt/1e7)
    #theta_plus = 0.05
    #_V_theta = V_theta * theta_decay + S * theta_plus
    theta_decay = jnp.exp(-dt/tau_theta)
    _v_theta = v_theta * theta_decay + s * theta_plus
    #_V_theta = V_theta + -V_theta * (dt/tau_theta) + S * alpha
    return _v_theta

class LIFCell(JaxComponent): ## leaky integrate-and-fire cell
    """
    A spiking cell based on leaky integrate-and-fire (LIF) neuronal dynamics.

    The specific differential equation that characterizes this cell
    is (for adjusting v, given current j, over time) is:

    | tau_m * dv/dt = (v_rest - v) + j * R
    | where R is the membrane resistance and v_rest is the resting potential
    | also, if a spike occurs, v is set to v_reset

    | --- Cell Input Compartments: ---
    | j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | rfr - (relative) refractory variable state
    | thr_theta - homeostatic/adaptive threshold increment state
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | s_raw - raw spike signals before post-processing (only if one_spike = True, else s_raw = s)
    | tols - time-of-last-spike

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        resist_m: membrane resistance value (Default: 1)

        thr: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV)

        v_rest: membrane resting potential (in mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value

        v_decay: decay factor applied to voltage leak (Default: 1.); setting this
            to 0 mV recovers pure integrate-and-fire (IF) dynamics

        tau_theta: homeostatic threshold time constant

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

        refract_time: relative refractory period time (ms; Default: 1 ms)

        one_spike: if True, a single-spike constraint will be enforced for
            every time step of neuronal dynamics simulated, i.e., at most, only
            a single spike will be permitted to emit per step -- this means that
            if > 1 spikes emitted, a single action potential will be randomly
            sampled from the non-zero spikes detected (Default: False)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuray of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)
    """

    # Define Functions
    def __init__(self, name, n_units, tau_m, resist_m=1., thr=-52., v_rest=-65.,
                 v_reset=-60., v_decay=1., tau_theta=1e7, theta_plus=0.05,
                 refract_time=5., thr_jitter=0., one_spike=False,
                 integration_type="euler", **kwargs):
        super().__init__(name, **kwargs)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.R_m = resist_m ## resistance value
        self.one_spike = one_spike ## True => constrains system to simulate 1 spike per time step

        self.v_rest = v_rest #-65. # mV
        self.v_reset = v_reset # -60. # -65. # mV (milli-volts)
        self.v_decay = v_decay ## controls strength of voltage leak (1 -> LIF, 0 => IF)
        ## basic asserts to prevent neuronal dynamics breaking...
        #assert (self.v_decay * self.dt / self.tau_m) <= 1. ## <-- to integrate in verify...
        assert self.R_m > 0.
        self.tau_theta = tau_theta ## threshold time constant # ms (0 turns off)
        self.theta_plus = theta_plus #0.05 ## threshold increment
        self.refract_T = refract_time #5. # 2. ## refractory period  # ms
        self.thr = thr ## (fixed) base value for threshold  #-52 # -72. # mV

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        ## set up surrogate function for spike emission
        self.spike_fx, self.d_spike_fx = secant_lif_estimator()
        #self.spike_fx, self.d_spike_fx = arctan_estimator() #
        #self.spike_fx, self.d_spike_fx = triangular_estimator() # straight_through_estimator()

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        thr0 = 0.
        if thr_jitter > 0.:
            key, subkey = random.split(self.key.value)
            thr0 = random.uniform(subkey, (1, n_units), minval=-thr_jitter,
                                  maxval=thr_jitter, dtype=jnp.float32)
        self.j = Compartment(restVals)
        self.v = Compartment(restVals + self.v_rest)
        self.s = Compartment(restVals)
        self.s_raw = Compartment(restVals)
        self.rfr = Compartment(restVals + self.refract_T)
        self.thr_theta = Compartment(restVals + thr0)
        self.tols = Compartment(restVals) ## time-of-last-spike
        self.surrogate = Compartment(restVals + 1.)  ## surrogate signal

    @staticmethod
    def _advance_state(t, dt, tau_m, R_m, v_rest, v_reset, v_decay, refract_T,
                       thr, tau_theta, theta_plus, one_spike, intgFlag, d_spike_fx,
                       key, j, v, s, rfr, thr_theta, tols):
        skey = None ## this is an empty dkey if single_spike mode turned off
        if one_spike:
            key, skey = random.split(key, 2)
        ## run one integration step for neuronal dynamics
        j = j * R_m
        #surrogate = d_spike_fx(v, thr + thr_theta)
        v, s, raw_spikes, rfr = _run_cell(dt, j, v, thr, thr_theta, rfr, skey,
                                          tau_m, v_rest, v_reset, v_decay, refract_T,
                                          intgFlag)
        surrogate = d_spike_fx(v, thr + thr_theta)
        #surrogate = d_spike_fx(j, thr + thr_theta)
        if tau_theta > 0.:
            ## run one integration step for threshold dynamics
            thr_theta = _update_theta(dt, thr_theta, raw_spikes, tau_theta, theta_plus)
        ## update tols
        tols = _update_times(t, s, tols)
        return v, s, raw_spikes, rfr, thr_theta, tols, key, surrogate

    @resolver(_advance_state)
    def advance_state(self, v, s, s_raw, rfr, thr_theta, tols, key, surrogate):
        self.v.set(v)
        self.s.set(s)
        self.s_raw.set(s_raw)
        self.rfr.set(rfr)
        self.thr_theta.set(thr_theta)
        self.tols.set(tols)
        self.key.set(key)
        self.surrogate.set(surrogate)

    @staticmethod
    def _reset(batch_size, n_units, v_rest, refract_T):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals #+ 0
        v = restVals + v_rest
        s = restVals #+ 0
        s_raw = restVals
        rfr = restVals + refract_T
        #thr_theta = restVals ## do not reset thr_theta
        tols = restVals #+ 0
        surrogate = restVals + 1.
        return j, v, s, s_raw, rfr, tols, surrogate

    @resolver(_reset)
    def reset(self, j, v, s, s_raw, rfr, tols, surrogate):
        self.j.set(j)
        self.v.set(v)
        self.s.set(s)
        self.s_raw.set(s_raw)
        self.rfr.set(rfr)
        #self.thr_theta.set(thr_theta)
        self.tols.set(tols)
        self.surrogate.set(surrogate)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name,
                  threshold_theta=self.thr_theta.value,
                  key=self.key.value)

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.thr_theta.set( data['threshold_theta'] )
        if seeded == True:
            self.key.set( data['key'] )

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "LIFCell - evolves neurons according to leaky integrate-"
                         "and-fire spiking dynamics."
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
                 "tols": "Time-of-last-spike"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "tau_m": "Cell membrane time constant",
            "resist_m": "Membrane resistance value",
            "thr": "Base voltage threshold value",
            "v_rest": "Resting membrane potential value",
            "v_reset": "Reset membrane potential value",
            "v_decay": "Voltage leak/decay factor",
            "tau_theta": "Threshold/homoestatic increment time constant",
            "theta_plus": "Amount to increment threshold by upon occurrence of spike",
            "refract_time": "Length of relative refractory period (ms)",
            "thr_jitter": "Scale of random uniform noise to apply to initial condition of threshold",
            "one_spike": "Should only one spike be sampled/allowed to emit at any given time step?",
            "integration_type": "Type of numerical integration to use for the cell dynamics"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = (v_rest - v) + j * resist_m",
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
        X = LIFCell("X", 9, 0.0004, 3)
    print(X)
