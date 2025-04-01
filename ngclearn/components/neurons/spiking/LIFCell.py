from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit, nn
from functools import partial
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
from ngcsimlib.logger import info, warn
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2
from ngclearn.utils.surrogate_fx import (secant_lif_estimator, arctan_estimator,
                                         triangular_estimator,
                                         straight_through_estimator)

from ngcsimlib.compilers.process import transition
#from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

#@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_decay=1.): ## raw voltage dynamics
    mask = (rfr >= refract_T) * 1. # get refractory mask
    ## update voltage / membrane potential
    dv_dt = (v_rest - v) * v_decay + (j * mask)
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T, v_rest, v_decay = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T, v_rest, v_decay)
    return dv_dt

#@partial(jit, static_argnums=[3, 4])
def _update_theta(dt, v_theta, s, tau_theta, theta_plus=0.05):
    ### Runs homeostatic threshold update dynamics one step (via Euler integration).
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
            plasticity (in milliVolts, or mV; default: -52. mV)

        v_rest: membrane resting potential (in mV; default: -65 mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value;
            (default: -60 mV)

        v_decay: decay factor applied to voltage leak (Default: 1.); setting this
            to 0 mV recovers pure integrate-and-fire (IF) dynamics

        tau_theta: homeostatic threshold time constant

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

        refract_time: relative refractory period time (ms; Default: 5 ms)

        one_spike: if True, a single-spike constraint will be enforced for
            every time step of neuronal dynamics simulated, i.e., at most, only
            a single spike will be permitted to emit per step -- this means that
            if > 1 spikes emitted, a single action potential will be randomly
            sampled from the non-zero spikes detected (Default: False)

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

        lower_clamp_voltage: if True, this will ensure voltage never is below
            the value of `v_rest` (default: True)
    """ ## batch_size arg?

    @deprecate_args(thr_jitter=None)
    def __init__(self, name, n_units, tau_m, resist_m=1., thr=-52., v_rest=-65.,
                 v_reset=-60., v_decay=1., tau_theta=1e7, theta_plus=0.05,
                 refract_time=5., one_spike=False, integration_type="euler",
                 surrogate_type="straight_through", lower_clamp_voltage=True,
                 **kwargs):
        super().__init__(name, **kwargs)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.resist_m = resist_m ## resistance value
        self.one_spike = one_spike ## True => constrains system to simulate 1 spike per time step
        self.lower_clamp_voltage = lower_clamp_voltage ## True ==> ensures voltage is never < v_rest

        self.v_rest = v_rest #-65. # mV
        self.v_reset = v_reset # -60. # -65. # mV (milli-volts)
        self.v_decay = v_decay ## controls strength of voltage leak (1 -> LIF, 0 => IF)
        ## basic asserts to prevent neuronal dynamics breaking...
        #assert (self.v_decay * self.dt / self.tau_m) <= 1. ## <-- to integrate in verify...
        assert self.resist_m > 0.
        self.tau_theta = tau_theta ## threshold time constant # ms (0 turns off)
        self.theta_plus = theta_plus #0.05 ## threshold increment
        self.refract_T = refract_time #5. # 2. ## refractory period  # ms
        self.thr = thr ## (fixed) base value for threshold  #-52 # -72. # mV

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        ## set up surrogate function for spike emission
        if surrogate_type == "secant_lif":
            self.spike_fx, self.d_spike_fx = secant_lif_estimator()
        elif surrogate_type == "arctan":
            self.spike_fx, self.d_spike_fx = arctan_estimator()
        elif surrogate_type == "triangular":
            self.spike_fx, self.d_spike_fx = triangular_estimator()
        else: ## default: straight_through
            self.spike_fx, self.d_spike_fx = straight_through_estimator()


        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals, display_name="Current", units="mA")
        self.v = Compartment(restVals + self.v_rest,
                             display_name="Voltage", units="mV")
        self.s = Compartment(restVals, display_name="Spikes")
        self.s_raw = Compartment(restVals, display_name="Raw Spike Pulses")
        self.rfr = Compartment(restVals + self.refract_T,
                               display_name="Refractory Time Period", units="ms")
        self.thr_theta = Compartment(restVals, display_name="Threshold Adaptive Shift",
                                     units="mV")
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike",
                                units="ms") ## time-of-last-spike
        self.surrogate = Compartment(restVals + 1., display_name="Surrogate State Value")

    @transition(output_compartments=["v", "s", "s_raw", "rfr", "thr_theta", "tols", "key", "surrogate"])    
    @staticmethod
    def advance_state(
            t, dt, tau_m, resist_m, v_rest, v_reset, v_decay, refract_T, thr, tau_theta, theta_plus, 
            one_spike, lower_clamp_voltage, intgFlag, d_spike_fx, key, j, v, rfr, thr_theta, tols
    ):
        skey = None ## this is an empty dkey if single_spike mode turned off
        if one_spike:
            key, skey = random.split(key, 2)
        ## run one integration step for neuronal dynamics
        j = j * resist_m
        ############################################################################
        ### Runs leaky integrator (leaky integrate-and-fire; LIF) neuronal dynamics.
        _v_thr = thr_theta + thr ## calc present voltage threshold
        #mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
        ## update voltage / membrane potential
        v_params = (j, rfr, tau_m, refract_T, v_rest, v_decay)
        if intgFlag == 1:
            _, _v = step_rk2(0., v, _dfv, dt, v_params)
        else:
            _, _v = step_euler(0., v, _dfv, dt, v_params)
        ## obtain action potentials/spikes/pulses
        s = (_v > _v_thr) * 1.
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
        raw_spikes = raw_s
        v = _v
        rfr = _rfr

        surrogate = d_spike_fx(v, _v_thr) #d_spike_fx(v, thr + thr_theta)
        if tau_theta > 0.:
            ## run one integration step for threshold dynamics
            thr_theta = _update_theta(dt, thr_theta, raw_spikes, tau_theta, theta_plus)
        ## update tols
        tols = (1. - s) * tols + (s * t)
        if lower_clamp_voltage: ## ensure voltage never < v_rest
            v = jnp.maximum(v, v_rest)
        return v, s, raw_spikes, rfr, thr_theta, tols, key, surrogate

    @transition(output_compartments=["j", "v", "s", "s_raw", "rfr", "tols", "surrogate"])
    @staticmethod
    def reset(batch_size, n_units, v_rest, refract_T):
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

    def save(self, directory, **kwargs):
        ## do a protected save of constants, depending on whether they are floats or arrays
        tau_m = (self.tau_m if isinstance(self.tau_m, float)
                 else jnp.asarray([[self.tau_m * 1.]]))
        thr = (self.thr if isinstance(self.thr, float)
               else jnp.asarray([[self.thr * 1.]]))
        v_rest = (self.v_rest if isinstance(self.v_rest, float)
                  else jnp.asarray([[self.v_rest * 1.]]))
        v_reset = (self.v_reset if isinstance(self.v_reset, float)
                   else jnp.asarray([[self.v_reset * 1.]]))
        v_decay = (self.v_decay if isinstance(self.v_decay, float)
                   else jnp.asarray([[self.v_decay * 1.]]))
        resist_m = (self.resist_m if isinstance(self.resist_m, float)
                    else jnp.asarray([[self.resist_m * 1.]]))
        tau_theta = (self.tau_theta if isinstance(self.tau_theta, float)
                     else jnp.asarray([[self.tau_theta * 1.]]))
        theta_plus = (self.theta_plus if isinstance(self.theta_plus, float)
                      else jnp.asarray([[self.theta_plus * 1.]]))

        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name,
                  threshold_theta=self.thr_theta.value,
                  tau_m=tau_m, thr=thr, v_rest=v_rest,
                  v_reset=v_reset, v_decay=v_decay,
                  resist_m=resist_m, tau_theta=tau_theta,
                  theta_plus=theta_plus,
                  key=self.key.value)

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.thr_theta.set(data['threshold_theta'])
        ## constants loaded in
        self.tau_m = data['tau_m']
        self.thr = data['thr']
        self.v_rest = data['v_rest']
        self.v_reset = data['v_reset']
        self.v_decay = data['v_decay']
        self.resist_m = data['resist_m']
        self.tau_theta = data['tau_theta']
        self.theta_plus = data['theta_plus']

        if seeded:
            self.key.set(data['key'])

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
            "theta_plus": "Amount to increment threshold by upon occurrence "
                          "of spike",
            "refract_time": "Length of relative refractory period (ms)",
            "one_spike": "Should only one spike be sampled/allowed to emit at "
                         "any given time step?",
            "integration_type": "Type of numerical integration to use for the "
                                "cell dynamics",
            "surrgoate_type": "Type of surrogate function to use approximate "
                              "derivative of spike w.r.t. voltage/current",
            "lower_bound_clamp": "Should voltage be lower bounded to be never "
                                 "be below `v_rest`"
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
