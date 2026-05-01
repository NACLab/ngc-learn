from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit, nn, Array
from ngcsimlib import deprecate_args
from ngcsimlib.logger import info, warn
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2
# from ngclearn.utils.surrogate_fx import (secant_lif_estimator, arctan_estimator,
#                                          triangular_estimator,
#                                          straight_through_estimator)

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

from ngclearn.components.neurons.spiking.LIFCell import LIFCell

@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_c, a0): ## raw voltage dynamics
    mask = (rfr >= refract_T) * 1. # get refractory mask
    ## update voltage / membrane potential
    dv_dt = ((v_rest - v) * (v - v_c) * a0) + (j * mask)
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T, v_rest, v_c, a0 = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_c, a0)
    return dv_dt

#@partial(jit, static_argnums=[3, 4])
def _update_theta(dt, v_theta, s, tau_theta, theta_plus: Array=0.05):
    ### Runs homeostatic threshold update dynamics one step (via Euler integration).
    #theta_decay = 0.9999999 #0.999999762 #jnp.exp(-dt/1e7)
    #theta_plus = 0.05
    #_V_theta = V_theta * theta_decay + S * theta_plus
    theta_decay = jnp.exp(-dt/tau_theta)
    _v_theta = v_theta * theta_decay + s * theta_plus
    #_V_theta = V_theta + -V_theta * (dt/tau_theta) + S * alpha
    return _v_theta

class QuadLIFCell(LIFCell): ## quadratic integrate-and-fire cell
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

        critical_v: critical voltage value (in mV) (i.e., variable name - a0)

        tau_theta: homeostatic threshold time constant 

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

        refract_time: relative refractory period time (ms; Default: 1 ms)

        one_spike: if True, a single-spike constraint will be enforced for
            every time step of neuronal dynamics simulated, i.e., at most, only
            a single spike will be permitted to emit per step -- this means that
            if > 1 spikes emitted, a single action potential will be randomly
            sampled from the non-zero spikes detected
            
        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuracy of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)

        surrogate_type: type of surrogate function to use for approximating a
            partial derivative of this cell's spikes w.r.t. its voltage/current
            (default: "straight_through")

            :Note: surrogate options available include: "straight_through"
                (straight-through estimator), "triangular" (triangular estimator),
                "arctan" (arc-tangent estimator), and "secant_lif" (the
                LIF-specialized secant estimator)
                
        v_min: minimum voltage to clamp dynamics to (Default: None)
    """ ## batch_size arg?

    @deprecate_args(thr_jitter=None, critical_V="critical_v")
    def __init__(
            self, name, n_units, tau_m, resist_m=1., thr=-52., v_rest=-65., v_reset=-60., v_scale=-41.6, critical_v=1.,
            tau_theta=1e7, theta_plus=0.05, refract_time=5., one_spike=False, integration_type="euler",
            surrogate_type="straight_through", v_min=None, **kwargs
    ):
        super().__init__(
            name, n_units, tau_m, resist_m, thr, v_rest, v_reset, 1., tau_theta, theta_plus, refract_time,
            one_spike, integration_type, surrogate_type, v_min=v_min, **kwargs
        )

        ## only two distinct additional constants distinguish the Quad-LIF cell
        self.v_c = v_scale
        self.a0 = critical_v

    @compilable
    def advance_state(self, dt, t):
        j = self.j.get() * self.resist_m

        _v_thr = self.thr_theta.get() + self.thr  ## calc present voltage threshold

        v_params = (j, self.rfr.get(), self.tau_m, self.refract_T, self.v_rest, self.v_c, self.a0)

        if self.intgFlag == 1:
            _, _v = step_rk2(0., self.v.get(), _dfv, dt, v_params)
        else:
            _, _v = step_euler(0., self.v.get(), _dfv, dt, v_params)

        s = (_v > _v_thr) * 1.
        _rfr = (self.rfr.get() + dt) * (1. - s)
        _v = _v * (1. - s) + s * self.v_reset

        raw_s = s

        #surrogate = d_spike_fx(v, _v_thr)  # d_spike_fx(v, thr + thr_theta)

        if self.one_spike and not self.max_one_spike:
            key, skey = random.split(self.key.get(), 2)

            m_switch = (jnp.sum(s) > 0.).astype(jnp.float32)  ## TODO: not batch-able
            rS = s * random.uniform(skey, s.shape)
            rS = nn.one_hot(jnp.argmax(rS, axis=1), num_classes=s.shape[1],
                            dtype=jnp.float32)
            s = s * (1. - m_switch) + rS * m_switch
            self.key.set(key)

        if self.max_one_spike:
            rS = nn.one_hot(jnp.argmax(self.v.get(), axis=1), num_classes=s.shape[1],
                            dtype=jnp.float32)  ## get max-volt spike
            s = s * rS  ## mask out non-max volt spikes

        if self.tau_theta > 0.:
            ## run one integration step for threshold dynamics
            thr_theta = _update_theta(dt, self.thr_theta.get(), raw_s, self.tau_theta, self.theta_plus)  # .get())
            self.thr_theta.set(thr_theta)

        ## update tols
        self.tols.set((1. - s) * self.tols.get() + (s * t))

        if self.v_min is not None:  ## ensures voltage never < v_rest
            _v = jnp.maximum(_v, self.v_min)

        self.v.set(_v)
        self.s.set(s)
        self.s_raw.set(raw_s)
        self.rfr.set(_rfr)

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size, self.n_units))
        if not self.j.targeted:
            self.j.set(restVals)
        self.v.set(restVals + self.v_rest)
        self.s.set(restVals)
        self.s_raw.set(restVals)
        self.rfr.set(restVals + self.refract_T)
        self.tols.set(restVals)
        #self.surrogate.set(restVals)

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
                 "thr_theta": "Current state of homeostatic adaptive threshold at time t",
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
            "theta_plus": "Amount to increment threshold by upon occurrence of a spike",
            "refract_time": "Length of relative refractory period (ms)",
            "one_spike": "Should only one spike be sampled/allowed to emit at any given time step?",
            "integration_type": "Type of numerical integration to use for the cell dynamics",
            "surrgoate_type": "Type of surrogate function to use approximate "
                              "derivative of spike w.r.t. voltage/current",
            "v_min": "Minimum voltage allowed before voltage variables are min-clipped/clamped"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = (v_rest - v) + j * resist_m",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = QuadLIFCell("X", 9, 0.0004, 3)
    print(X)
