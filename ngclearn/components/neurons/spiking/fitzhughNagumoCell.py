from ngcsimlib.component import Component
from jax import numpy as jnp, random, jit
from functools import partial
import time
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

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
def _dfv_internal(j, v, w, a, b, g, tau_m): ## raw voltage dynamics
    dv_dt = v - jnp.power(v, 3)/g - w + j ## dv/dt
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, w, a, b, g, tau_m = params
    dv_dt = _dfv_internal(j, v, w, a, b, g, tau_m)
    return dv_dt

@jit
def _dfw_internal(j, v, w, a, b, g, tau_w): ## raw recovery dynamics
    dw_dt = v + a - b * w ## dw/dt
    dw_dt = dw_dt * (1./tau_w)
    return dw_dt

def _dfw(t, w, params): ## recovery dynamics wrapper
    j, v, a, b, g, tau_m = params
    dv_dt = _dfw_internal(j, v, w, a, b, g, tau_m)
    return dv_dt

@jit
def _emit_spike(v, v_thr):
    s = (v > v_thr).astype(jnp.float32)
    return s

#@partial(jit, static_argnums=[10])
def run_cell(dt, j, v, w, v_thr, tau_m, tau_w, a, b, g=3., integType=0):
    """

    Args:
        dt: integration time constant

        j: electrical current

        v: membrane potential / voltage

        w: recovery variable value(s)

        v_thr: voltage/membrane threshold (to obtain action potentials in terms
            of binary spikes)

        tau_m: membrane time constant

        tau_w: recover variable time constant (Default: 12.5 ms)

        a: dimensionless recovery variable shift factor "alpha" (Default: 0.7)

        b: dimensionless recovery variable scale factor "beta" (Default: 0.8)

        g: power-term divisor 'gamma' (Default: 3.)

        integType: integration type to use (0 --> Euler/RK1, 1 --> Midpoint/RK2)

    Returns:
        updated voltage, updated recovery, spikes
    """
    if integType == 1:
        v_params = (j, w, a, b, g, tau_m)
        _, _v = step_rk2(0., v, _dfv, dt, v_params) #_v = step_rk2(v, v_params, _dfv, dt)
        w_params = (j, v, a, b, g, tau_w)
        _, _w = step_rk2(0., w, _dfw, dt, w_params) #_w = step_rk2(w, w_params, _dfw, dt)
    else: # integType == 0 (default -- Euler)
        v_params = (j, w, a, b, g, tau_m)
        _, _v = step_euler(0., v, _dfv, dt, v_params) #_v = step_euler(v, v_params, _dfv, dt)
        w_params = (j, v, a, b, g, tau_w)
        _, _w = step_euler(0., w, _dfw, dt, w_params) #_w = step_euler(w, w_params, _dfw, dt)
    #s = (_v > v_thr).astype(jnp.float32)
    s = _emit_spike(_v, v_thr)
    return _v, _w, s

class FitzhughNagumoCell(Component):
    """
    The Fitzhugh-Nagumo neuronal cell model; a two-variable simplification
    of the Hodgkin-Huxley (squid axon) model. This cell model iteratively evolves
    voltage "v" and recovery "w" (which represents the combined effects of
    sodium channel deinactivation and potassium channel deactivation in the
    Hodgkin-Huxley model).

    The specific pair of differential equations that characterize this cell
    are (for adjusting v and w, given current j, over time):

    | tau_m * dv/dt = v - (v^3)/3 - w + j
    | tau_w * dw/dt = v + a - b * w

    | References:
    | FitzHugh, Richard. "Impulses and physiological states in theoretical
    | models of nerve membrane." Biophysical journal 1.6 (1961): 445-466.
    |
    | Nagumo, Jinichi, Suguru Arimoto, and Shuji Yoshizawa. "An active pulse
    | transmission line simulating nerve axon." Proceedings of the IRE 50.10
    | (1962): 2061-2070.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        tau_w: recover variable time constant (Default: 12.5 ms)

        alpha: dimensionless recovery variable shift factor "a" (Default: 0.7)

        beta: dimensionless recovery variable scale factor "b" (Default: 0.8)

        gamma: power-term divisor (Default: 3.)

        v_thr: voltage/membrane threshold (to obtain action potentials in terms
            of binary spikes)

        v0: initial condition / reset for voltage

        w0: initial condition / reset for recovery

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuray of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)

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
    def voltageName(cls):
        return 'v'

    @classmethod
    def recoveryName(cls):
        return 'w'

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
    def voltage(self):
        return self.compartments.get(self.voltageName(), None)

    @voltage.setter
    def voltage(self, t):
        self.compartments[self.voltageName()] = t

    @property
    def recovery(self):
        return self.compartments.get(self.recoveryName(), None)

    @recovery.setter
    def recovery(self, t):
        self.compartments[self.recoveryName()] = t

    @property
    def timeOfLastSpike(self):
        return self.compartments.get(self.timeOfLastSpikeCompartmentName(), None)

    @timeOfLastSpike.setter
    def timeOfLastSpike(self, t):
        self.compartments[self.timeOfLastSpikeCompartmentName()] = t

    # Define Functions
    def __init__(self, name, n_units, tau_m=1., tau_w=12.5, alpha=0.7,
                 beta=0.8, gamma=3., v_thr=1.07, v0=0., w0=0.,
                 integration_type="euler", key=None, useVerboseDict=False,
                 **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ## Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Cell properties
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.v0 = v0 ## initial membrane potential/voltage condition
        self.w0 = w0 ## initial w-parameter condition
        self.v_thr = v_thr

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units
        self.reset()

    def verify_connections(self):
        pass

    def advance_state(self, t, dt, **kwargs):
        self.key, *subkeys = random.split(self.key, 2)

        j = self.inputCompartment
        v = self.voltage
        w = self.recovery

        v, w, s = run_cell(dt, j, v, w, self.v_thr, self.tau_m, self.tau_w, self.alpha,
                           self.beta, self.gamma, self.intgFlag)

        self.voltage = v
        self.recovery = w
        self.outputCompartment = s
        self.timeOfLastSpike = update_times(t, self.outputCompartment, self.timeOfLastSpike)

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.voltage = jnp.zeros((self.batch_size, self.n_units)) + self.v0
        self.recovery = jnp.zeros((self.batch_size, self.n_units)) + self.w0
        self.outputCompartment = jnp.zeros((self.batch_size, self.n_units)) #None
        self.timeOfLastSpike = jnp.zeros((self.batch_size, self.n_units))

    def save(self, **kwargs):
        pass
