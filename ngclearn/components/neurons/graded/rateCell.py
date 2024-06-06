from jax import numpy as jnp, random, jit
from functools import partial
import time, math
from ngclearn.utils import tensorstats
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.model_utils import create_function, threshold_soft, \
                                       threshold_cauchy
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

## rewritten code
@partial(jit, static_argnums=[3, 4, 5])
def _dfz_internal(z, j, j_td, tau_m, leak_gamma, prior_type=None): ## raw dynamics
    z_leak = z # * 2 ## Default: assume Gaussian
    if prior_type != None:
        if prior_type == "laplacian": ## Laplace dist
            z_leak = jnp.sign(z) ## d/dx of Laplace is signum
        elif prior_type == "cauchy":  ## Cauchy dist: x ~ (1.0 + tf.math.square(z))
            z_leak = (z * 2)/(1. + jnp.square(z))
        elif prior_type == "exp":  ## Exp dist: x ~ -exp(-x^2)
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

def run_cell(dt, j, j_td, z, tau_m, leak_gamma=0., integType=0, priorType=None):
    """
    Runs leaky rate-coded state dynamics one step in time.

    Args:
        dt: integration time constant

        j: input (bottom-up) electrical/stimulus current

        j_td: modulatory (top-down) electrical/stimulus pressure

        z: current value of membrane/state

        tau_m: membrane/state time constant

        leak_gamma: strength of leak to apply to membrane/state

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

class RateCell(JaxComponent): ## Rate-coded/real-valued cell
    """
    A non-spiking cell driven by the gradient dynamics of neural generative
    coding-driven predictive processing.

    The specific differential equation that characterizes this cell
    is (for adjusting v, given current j, over time) is:

    | tau_m * dz/dt = lambda * prior(z) + (j + j_td)
    | where j is the set of general incoming input signals (e.g., message-passed signals)
    | and j_td is taken to be the set of top-down pressure signals

    | --- Cell Compartments: ---
    | j - input (takes in external signals)
    | j_td - input/top-down pressure input (takes in external signals)
    | z - rate activity
    | zF - post-activation function activity, i.e., fx(z)

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
    """

    # Define Functions
    def __init__(self, name, n_units, tau_m, prior=("gaussian", 0.), act_fx="identity",
                 threshold=("none", 0.), integration_type="euler", **kwargs):
        super().__init__(name, **kwargs)

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

        ## Layer size setup
        self.n_units = n_units
        self.batch_size = 1
        self.fx, self.dfx = create_function(fun_name=act_fx)

        # compartments (state of the cell & parameters will be updated through stateless calls)
        restVals = jnp.zeros((self.batch_size, n_units))
        self.j = Compartment(restVals) # electrical current
        self.zF = Compartment(restVals) # rate-coded output - activity
        self.j_td = Compartment(restVals) # top-down electrical current - pressure
        self.z = Compartment(restVals) # rate activity

    @staticmethod
    def _advance_state(t, dt, fx, dfx, tau_m, priorLeakRate, intgFlag, priorType,
                       thresholdType, thr_lmbda, j, j_td, z, zF):
        if tau_m > 0.:
            ### run a step of integration over neuronal dynamics
            ## Notes:
            ## self.pressure <-- "top-down" expectation / contextual pressure
            ## self.current <-- "bottom-up" data-dependent signal
            dfx_val = dfx(z)
            j = modulate(j, dfx_val)
            tmp_z = run_cell(dt, j, j_td, z,
                         tau_m, leak_gamma=priorLeakRate,
                         integType=intgFlag, priorType=priorType)
            ## apply optional thresholding sub-dynamics
            if thresholdType == "soft_threshold":
                tmp_z = threshold_soft(tmp_z, thr_lmbda)
            elif thresholdType == "cauchy_threshold":
                tmp_z = threshold_cauchy(tmp_z, thr_lmbda)
            z = tmp_z ## pre-activation function value(s)
            zF = fx(z) ## post-activation function value(s)
        else:
            ## run in "stateless" mode (when no membrane time constant provided)
            z = run_cell_stateless(j)
            zF = fx(z)
        return j, j_td, z, zF

    @resolver(_advance_state)
    def advance_state(self, j, j_td, z, zF):
        self.j.set(j)
        self.j_td.set(j_td)
        self.z.set(z)
        self.zF.set(zF)

    @staticmethod
    def _reset(batch_size, n_units):
        return tuple([jnp.zeros((batch_size, n_units)) for _ in range(4)])

    @resolver(_reset)
    def reset(self, j, zF, j_td, z):
        self.j.set(j) # electrical current
        self.zF.set(zF) # rate-coded output - activity
        self.j_td.set(j_td) # top-down electrical current - pressure
        self.z.set(z) # rate activity

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
        X = RateCell("X", 9, 0.03)
    print(X)
