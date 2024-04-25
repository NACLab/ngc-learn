from ngcsimlib.component import Component
from jax import numpy as jnp, random, jit, nn
from functools import partial
from ngclearn.utils.model_utils import create_function, threshold_soft, \
                                       threshold_cauchy
import time, sys
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

## rewritten code
@partial(jit, static_argnums=[3, 4, 5])
def _dfz_internal(z, j, j_td, tau_m, leak_gamma, priorType=None): ## raw dynamics
    z_leak = z # * 2 ## Default: assume Gaussian
    if priorType != None:
        if priorType == "laplacian": ## Laplace dist
            z_leak = jnp.sign(z) ## d/dx of Laplace is signum
        elif priorType == "cauchy":  ## Cauchy dist: x ~ (1.0 + tf.math.square(z))
            z_leak = (z * 2)/(1. + jnp.square(z))
        elif priorType == "exp":  ## Exp dist: x ~ -exp(-x^2)
            z_leak = jnp.exp(-jnp.square(z)) * z * 2
    dz_dt = (-z_leak * leak_gamma + (j + j_td)) * (1./tau_m)
    return dz_dt

def _dfz(t, z, params): ## diff-eq dynamics wrapper
    j, j_td, tau_m, leak_gamma, priorType = params
    dz_dt = _dfz_internal(z, j, j_td, tau_m, leak_gamma, priorType)
    return dz_dt

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

def run_cell(dt, j, j_td, z, tau_m, leak_gamma=0., beta=1., integType=0,
             priorType=None):
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

        priorType: scale-shift prior distribution to impose over neural dynamics

    Returns:
        New value of membrane/state for next time step
    """
    if integType == 1:
        params = (j, j_td, tau_m, leak_gamma, priorType)
        _, _z = step_rk2(0., z, _dfz, dt, params)
    else:
        params = (j, j_td, tau_m, leak_gamma, priorType)
        _, _z = step_euler(0., z, _dfz, dt, params)
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

        prior: a kernel for specifying the type of centered scale-shift distribution
            to impose over neuronal dynamics, applied to each neuron or
            dimension within this component (Default: ("gaussian", 0)); this is
            a tuple with 1st element containing a string name of the distribution
            one wants to use while the second value is a `leak rate` scalar
            that controls the influence/weighting that this distribution
            has on the dynamics; for example, ("laplacian, 0.001") means that a
            centered laplacian distribution scaled by `0.001` will be injected
            into this cell's dynamics ODE each step of simulated time

            :Note: supported scale-shift distributions include "laplacian",
                "cauchy", "exp", and "gaussian"

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
    def __init__(self, name, n_units, tau_m, prior=("gaussian", 0.),
                 act_fx="identity", threshold=("none", 0.),
                 integration_type="euler", key=None, useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant -- setting to 0 triggers "stateless" mode
        priorType, leakRate = prior
        self.priorType = priorType ## type of scale-shift prior to impose over the leak
        self.priorLeakRate = leakRate ## degree to which rate neurons leak (according to prior)
        thresholdType, thr_lmbda = threshold
        self.thresholdType = thresholdType ## type of thresholding function to use
        self.thr_lmbda = thr_lmbda ## scale to drive thresholding dynamics

        ## integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1
        self.fx, self.dfx = create_function(fun_name=act_fx)
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        if self.tau_m > 0.:
            ### run a step of integration over neuronal dynamics
            ## Notes:
            ## self.pressure <-- "top-down" expectation / contextual pressure
            ## self.current <-- "bottom-up" data-dependent signal
            dfx_val = self.dfx(self.rateActivity)
            self.current = modulate(self.current, dfx_val)
            z = run_cell(dt, self.current, self.pressure, self.rateActivity,
                         self.tau_m, leak_gamma=self.priorLeakRate,
                         integType=self.intgFlag, priorType=self.priorType)
            ## apply optional thresholding sub-dynamics
            if self.thresholdType == "soft_threshold":
                z = threshold_soft(z, self.thr_lmbda)
            elif self.thresholdType == "cauchy_threshold":
                z = threshold_cauchy(z, self.thr_lmbda)
            self.rateActivity = z
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
