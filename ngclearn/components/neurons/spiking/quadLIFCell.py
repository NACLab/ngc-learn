from jax import numpy as jnp, random, jit, nn
from functools import partial
import time, sys
from ngclearn.utils import tensorstats
from ngclearn import resolver, Component, Compartment
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2
## import parent cell class/component
from ngclearn.components.neurons.spiking.LIFCell import LIFCell

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

@jit
def _modify_current(j, dt, tau_m): ## electrical current re-scaling co-routine
    jScale = tau_m/dt
    return j * jScale

@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_c, a0): ## raw voltage dynamics
    mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    ## update voltage / membrane potential
    dv_dt = ((v_rest - v) * (v - v_c) * a0) + (j * mask)
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T, v_rest, v_c, a0 = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_c, a0)
    return dv_dt

#@partial(jit, static_argnums=[7,8,9,10,11,12,13,14])
def _run_cell(dt, j, v, v_thr, v_theta, rfr, skey, v_c, a0, tau_m, v_rest,
              v_reset, refract_T, integType=0):
    """
    Runs quadratic leaky integrator neuronal dynamics

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

        v_c: scaling factor for voltage accumulation

        a0: critical voltage value

        tau_m: cell membrane time constant

        v_rest: membrane resting potential (in mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value

        refract_T: (relative) refractory time period (in ms; Default
            value is 1 ms)

        integType: integer indicating type of integration to use

    Returns:
        voltage(t+dt), spikes, raw spikes, updated refactory variables
    """
    _v_thr = v_theta + v_thr ## calc present voltage threshold
    #mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    ## update voltage / membrane potential (v_c ~> 0.8?) (a0 usually <1?)
    #_v = v + ((v_rest - v) * (v - v_c) * a0) * (dt/tau_m) + (j * mask)
    v_params = (j, rfr, tau_m, refract_T, v_rest, v_c, a0)
    if integType == 1:
        _, _v = step_rk2(0., v, _dfv, dt, v_params)
    else:
        _, _v = step_euler(0., v, _dfv, dt, v_params)
    ## obtain action potentials
    s = (_v > _v_thr).astype(jnp.float32)
    ## update refractory variables
    _rfr = (rfr + dt) * (1. - s)
    ## perform hyper-polarization of neuronal cells
    _v = _v * (1. - s) + s * v_reset

    raw_s = s + 0 ## preserve un-altered spikes
    ############################################################################
    ## this is a spike post-processing step
    if skey is not None: ## FIXME: this would not work for mini-batches!!!!!!!
        m_switch = (jnp.sum(s) > 0.).astype(jnp.float32)
        rS = random.choice(skey, s.shape[1], p=jnp.squeeze(s))
        rS = nn.one_hot(rS, num_classes=s.shape[1], dtype=jnp.float32)
        s = s * (1. - m_switch) + rS * m_switch
    ############################################################################
    return _v, s, raw_s, _rfr

@partial(jit, static_argnums=[3,4])
def _update_theta(dt, v_theta, s, tau_theta, theta_plus=0.05):
    """
    Runs homeostatic threshold update dynamics one step.

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
    theta_decay = jnp.exp(-dt/tau_theta)
    _v_theta = v_theta * theta_decay + s * theta_plus
    return _v_theta

class QuadLIFCell(LIFCell): ## quadratic (leaky) LIF cell; inherits from LIFCell
    """
    A spiking cell based on quadratic leaky integrate-and-fire (LIF) neuronal
    dynamics. Note that QuadLIFCell is a child of LIFCell and inherits its
    main set of routines, only overriding its dynamics in advance().

    Dynamics can be taken to be governed by the following ODE:

    | d.Vz/d.t = a0 * (V - V_rest) * (V - V_c) + Jz * R) * (dt/tau_mem)

    where:

    |   a0 - scaling factor for voltage accumulation
    |   V_c - critical voltage (value)

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

        resist_m: membrane resistance value

        thr: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV)

        v_rest: membrane resting potential (in mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value

        v_scale: scaling factor for voltage accumulation (v_c)

        critical_V: critical voltage value (a0)

        tau_theta: homeostatic threshold time constant

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

        refract_time: relative refractory period time (ms; Default: 1 ms)

        one_spike: if True, a single-spike constraint will be enforced for
            every time step of neuronal dynamics simulated, i.e., at most, only
            a single spike will be permitted to emit per step -- this means that
            if > 1 spikes emitted, a single action potential will be randomly
            sampled from the non-zero spikes detected
    """

    # Define Functions
    def __init__(self, name, n_units, tau_m, resist_m=1., thr=-52., v_rest=-65.,
                 v_reset=60., v_scale=-41.6, critical_V=1., tau_theta=1e7,
                 theta_plus=0.05, refract_time=5., thr_jitter=0., one_spike=False,
                 integration_type="euler", **kwargs):
        super().__init__(name, n_units, tau_m, resist_m, thr, v_rest, v_reset,
                         1., tau_theta, theta_plus, refract_time, thr_jitter,
                         one_spike, integration_type, **kwargs)
        ## only two distinct additional constants distinguish the Quad-LIF cell
        self.v_c = v_scale
        self.a0 = critical_V

    @staticmethod
    def _advance_state(t, dt, tau_m, R_m, v_rest, v_reset, refract_T, thr,
                       tau_theta, theta_plus, one_spike, v_c, a0, intgFlag, key,
                       j, v, s, rfr, thr_theta, tols):
        ## Note: this runs quadratic LIF neuronal dynamics but constrained to be
        ## similar to the general form of LIF dynamics
        skey = None ## this is an empty dkey if single_spike mode turned off
        if one_spike: ## old code ~> if self.one_spike is False:
            key, *subkeys = random.split(key, 2)
            skey = subkeys[0]
        ## run one integration step for neuronal dynamics
        j = j * R_m
        v, s, raw_spikes, rfr = _run_cell(dt, j, v, thr, thr_theta, rfr, skey,
                                          v_c, a0, tau_m, v_rest, v_reset,
                                          refract_T, intgFlag)
        if tau_theta > 0.:
            ## run one integration step for threshold dynamics
            thr_theta = _update_theta(dt, thr_theta, raw_spikes, tau_theta,
                                      theta_plus)
        ## update tols
        tols = _update_times(t, s, tols)
        return v, s, raw_spikes, rfr, thr_theta, tols, key

    @resolver(_advance_state)
    def advance_state(self, v, s, s_raw, rfr, thr_theta, tols, key):
        self.v.set(v)
        self.s.set(s)
        self.s_raw.set(s_raw)
        self.rfr.set(rfr)
        self.thr_theta.set(thr_theta)
        self.tols.set(tols)
        self.key.set(key)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "QuadLIFCell - evolves neurons according to quadratic "
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
            "v_scale": "Scaling factor for voltage accumulation",
            "critical_V": "Critical voltage value",
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
    # NOTE: VN: currently error in init function
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = QuadLIFCell("X", 1, 10.)
    print(X)
