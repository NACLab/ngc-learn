from ngcsimlib.component import Component
from jax import numpy as jnp, random, jit, nn
from functools import partial
from ngclearn.utils.model_utils import create_function
import time, sys

@jit
def modulate(j, dfx_val):
    """
    Apply a signal modulator to j (typically of the form of a derivative/dampening function)

    Args:
        j: current/stimulus value to modulate

        dfx_val: modulator signal

    Returns:
        modulated j value
    """
    return j * dfx_val

@partial(jit, static_argnums=[3])
def _dfz(z, j, j_td, leak_gamma):
    dz_dt = (-z * leak_gamma + (j + j_td))
    return dz_dt

@partial(jit, static_argnums=[4,5,6])
def _step_euler(dt, j, j_td, z, tau_m, leak_gamma=0., beta=1.):  ## perform step of Euler/RK-1
    dz_dt = _dfz(z, j, j_td, leak_gamma)
    _z = z * beta + dz_dt * (1./tau_m) * dt
    return _z

@partial(jit, static_argnums=[4,5,6])
def _step_midpoint(dt, j, j_td, z, tau_m, leak_gamma=0., beta=1.):  ## perform step of RK-2
    ## take initial Euler step
    _z = step_euler(dt/2., j, j_td, z, tau_m, leak_gamma, beta)
    ## take 2nd Euler step on projected value (midpoint step)
    dz_dt = _dfz(_z, j, j_td, leak_gamma)
    _z2 = z * beta + dz_dt * (1./tau_m) * dt
    return _z2

@partial(jit, static_argnums=[4,5,6,7])
def run_cell(dt, j, j_td, z, tau_m, leak_gamma=0., beta=1., integType=0):
    """
    Runs leaky rate-coded state dynamics one step in time.

    Args:
        dt: integration time constant

        j: input (bottom-up) electrical/stimulus current

        j_td: modulatory (top-down) electrical/stimulus pressure

        z: current value of membrane/state

        tau_m: membrane/state time constant

        leak_gamma: strength of leak to apply to membrane/state

        beta: dampening coefficient (Default: 1.)

        integType: integration type to use (0 --> Euler/RK1, 1 --> Midpoint/RK2)

    Returns:
        New value of membrane/state for next time step
    """
    if integType == 1:
        _z = _step_midpoint(dt, j, j_td, z, tau_m, leak_gamma, beta)
    else:
        _z = _step_euler(dt, j, j_td, z, tau_m, leak_gamma, beta)
    return _z

@jit
def run_cell_stateless(j):
    """
    A simplification of running a stateless set of dynamics over j (an identity
    functional form of dynamics).

    Args:
        j: stimulus to do nothing to

    Returns:
        the stimulus
    """
    return j + 0

class RateCell(Component): ## Rate-coded/real-valued cell
    """
    A non-spiking cell driven by the gradient dynamics of neural generative
    coding-driven predictive processing.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane/state time constant (milliseconds)

        leakRate: membrane/state leak value (Default: 0.)

        act_fx: string name of activation function/nonlinearity to use

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuray of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)

        key: PRNG Key to control determinism of any underlying random values
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'j' ## electrical current

    @classmethod
    def outputCompartmentName(cls): # OR: activityName()
        return 'zF' ## rate-coded output

    @classmethod
    def pressureName(cls):
        return 'j_td'

    @classmethod
    def rateActivityName(cls):
        return 'z'

    ## Bind Properties to Compartments for ease of use
    @property
    def inputCompartment(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, out):
        self.compartments[self.inputCompartmentName()] = out

    @property
    def outputCompartment(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @outputCompartment.setter
    def outputCompartment(self, out):
        self.compartments[self.outputCompartmentName()] = out

    @property
    def current(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @current.setter
    def current(self, inp):
        self.compartments[self.inputCompartmentName()] = inp

    @property
    def pressure(self):
        return self.compartments.get(self.pressureName(), None)

    @pressure.setter
    def pressure(self, inp):
        self.compartments[self.pressureName()] = inp

    @property
    def rateActivity(self):
        return self.compartments.get(self.rateActivityName(), None)

    @rateActivity.setter
    def rateActivity(self, out):
        self.compartments[self.rateActivityName()] = out

    @property
    def activity(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @activity.setter
    def activity(self, out):
        self.compartments[self.outputCompartmentName()] = out

    # Define Functions
    def __init__(self, name, n_units, tau_m, leakRate=0., act_fx="identity",
                 integration_type="euler", key=None, useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant -- setting to 0 triggers "stateless" mode
        self.leakRate = leakRate ## degree to which rate neurons leak

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = 0
        if self.integrationType == "midpoint" or self.integrationType == "rk2":
            self.intgFlag = 1

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

        self.fx, self.dfx = create_function(fun_name=act_fx)

        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        if self.tau_m > 0.:
            ### run one step of Euler integration over neuronal dynamics
            ## Notes:
            ## self.pressure <-- "top-down" expectation / contextual pressure
            ## self.current <-- "bottom-up" data-dependent signal
            dfx_val = self.dfx(self.rateActivity)
            self.current = modulate(self.current, dfx_val)
            self.rateActivity = run_cell(dt, self.current, self.pressure,
                                         self.rateActivity, self.tau_m,
                                         leak_gamma=self.leakRate,
                                         integType=self.intgFlag)
            self.activity = self.fx(self.rateActivity)
            self.current = None
        else:
            ## run in "stateless" mode (when no membrane time constant provided)
            self.rateActivity = run_cell_stateless(self.current)
            self.activity = self.fx(self.rateActivity)
            #self.current = None

    def reset(self, **kwargs):
        self.current = jnp.zeros((self.batch_size, self.n_units))
        self.pressure = jnp.zeros((self.batch_size, self.n_units))
        self.rateActivity = jnp.zeros((self.batch_size, self.n_units))
        self.activity = jnp.zeros((self.batch_size, self.n_units))

    def save(self, **kwargs):
        pass
