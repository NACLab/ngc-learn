from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, nn, Array
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2
from ngclearn.utils.surrogate_fx import (secant_lif_estimator, arctan_estimator,
                                         triangular_estimator,
                                         straight_through_estimator)

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T, v_rest, g_L = params
    mask = (rfr >= refract_T) * 1.  # get refractory mask
    ## update voltage / membrane potential
    dv_dt = (v_rest - v) * g_L + (j * mask)
    dv_dt = dv_dt * (1. / tau_m)
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

        v_rest: reversal potential or membrane resting potential (in mV; default: -65 mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value;
            (default: -60 mV)

        conduct_leak: leak conductance (g_L) value or decay factor applied to voltage leak 
            (Default: 1.); setting this to 0 mV recovers pure integrate-and-fire (IF) dynamics

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

        v_min: minimum voltage to clamp dynamics to (Default: None)
    """ ## batch_size arg?

    def __init__(
            self, name, n_units, tau_m, resist_m=1., thr=-52., v_rest=-65., v_reset=-60., conduct_leak=1., tau_theta=1e7,
            theta_plus=0.05, refract_time=5., one_spike=False, integration_type="euler", surrogate_type="straight_through",
            v_min=None, max_one_spike=False, key=None
    ):
        super().__init__(name, key)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)
        self.one_spike = one_spike ## True => constrains system to simulate 1 spike per time step
        self.max_one_spike = max_one_spike

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.resist_m = resist_m ## resistance value

        self.v_min = v_min ## ensures voltage is never < v_min

        self.v_rest = v_rest #-65. # mV
        self.v_reset = v_reset # -60. # -65. # mV (milli-volts)
        self.g_L = conduct_leak ## controls strength of voltage leak (1 -> LIF, 0 => IF)
        ## basic asserts to prevent neuronal dynamics breaking...
        #assert (self.conduct_leak * self.dt / self.tau_m) <= 1. ## <-- to integrate in verify...
        assert self.resist_m > 0.
        self.tau_theta = tau_theta ## threshold time constant # ms (0 turns off)
        self.theta_plus = theta_plus #0.05 ## threshold increment
        self.refract_T = refract_time #5. # 2. ## refractory period  # ms
        self.thr = thr ## (fixed) base value for threshold  #-52 # -72. # mV

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        # ## set up surrogate function for spike emission
        # if surrogate_type == "secant_lif":
        #     spike_fx, d_spike_fx = secant_lif_estimator()
        # elif surrogate_type == "arctan":
        #     spike_fx, d_spike_fx = arctan_estimator()
        # elif surrogate_type == "triangular":
        #     spike_fx, d_spike_fx = triangular_estimator()
        # else: ## default: straight_through
        #     spike_fx, d_spike_fx = straight_through_estimator()

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals, display_name="Current", units="mA")
        self.v = Compartment(restVals + self.v_rest, display_name="Voltage", units="mV")
        self.s = Compartment(restVals, display_name="Spikes")
        self.s_raw = Compartment(restVals, display_name="Raw Spike Pulses")
        self.rfr = Compartment(restVals + self.refract_T, display_name="Refractory Time Period", units="ms")
        self.thr_theta = Compartment(restVals, display_name="Threshold Adaptive Shift", units="mV")
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike", units="ms") ## time-of-last-spike
        # self.surrogate = Compartment(restVals + 1., display_name="Surrogate State Value")

    @compilable
    def advance_state(self, dt, t):
        j = self.j.get() * self.resist_m

        _v_thr = self.thr_theta.get() + self.thr  ## calc present voltage threshold

        v_params = (j, self.rfr.get(), self.tau_m.get(), self.refract_T, self.v_rest, self.g_L)

        if self.intgFlag == 1:
            _, _v = step_rk2(0., self.v.get(), _dfv, dt, v_params)
        else:
            _, _v = step_euler(0., self.v.get(), _dfv, dt, v_params)

        s = (_v > _v_thr) * 1.
        _rfr = (self.rfr.get() + dt) * (1. - s)
        _v = _v * (1. - s) + s * self.v_reset

        raw_s = s

        if self.one_spike and not self.max_one_spike:
            key, skey = random.split(self.key.get(), 2)

            m_switch = (jnp.sum(s) > 0.).astype(jnp.float32) ## TODO: not batch-able
            rS = s * random.uniform(skey, s.shape)
            rS = nn.one_hot(jnp.argmax(rS, axis=1), num_classes=s.shape[1], dtype=jnp.float32)
            s = s * (1. - m_switch) + rS * m_switch
            self.key.set(key)

        if self.max_one_spike:
            rS = nn.one_hot(jnp.argmax(self.v.get(), axis=1), num_classes=s.shape[1], dtype=jnp.float32) ## get max-volt spike
            s = s * rS ## mask out non-max volt spikes

        if self.tau_theta > 0.:
            ## run one integration step for threshold dynamics
            thr_theta = _update_theta(dt, self.thr_theta.get(), raw_s, self.tau_theta, self.theta_plus) #.get())
            self.thr_theta.set(thr_theta)

        ## update time-of-last spike variable(s)
        self.tols.set((1. - s) * self.tols.get() + (s * t))

        if self.v_min is not None: ## ensures voltage never < v_rest
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
            "conduct_leak": "Conductance leak / voltage decay factor",
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
        X = LIFCell("X", 9, 0.0004, 3)
    print(X)
