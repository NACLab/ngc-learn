from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit, nn
from ngcsimlib import deprecate_args
from ngcsimlib.logger import info, warn
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

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

class FitzhughNagumoCell(JaxComponent): ## F-H cell
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

    | --- Cell Input Compartments: ---
    | j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | w - recovery variable state
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | tols - time-of-last-spike

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

        resist_m: membrane resistance value

        tau_w: recover variable time constant (Default: 12.5 ms)

        alpha: dimensionless recovery variable shift factor "a" (Default: 0.7)

        beta: dimensionless recovery variable scale factor "b" (Default: 0.8)

        gamma: power-term divisor (Default: 3.)

        v0: initial condition / reset for voltage

        w0: initial condition / reset for recovery

        v_thr: voltage/membrane threshold (to obtain action potentials in terms
            of binary spikes)

        spike_reset: if True, once voltage crosses threshold, then dynamics
            of voltage and recovery are reset/snapped to initial conditions
            (default: False)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuracy of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)
    """

    def __init__(
            self, name, n_units, tau_m=1., resist_m=1., tau_w=12.5, alpha=0.7, beta=0.8, gamma=3., v0=0., w0=0.,
            v_thr=1.07, spike_reset=False, integration_type="euler", **kwargs
    ):
        super().__init__(name, **kwargs)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Cell properties
        self.tau_m = tau_m
        self.resist_m = resist_m ## resistance R_m
        self.tau_w = tau_w
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.v0 = v0 ## initial membrane potential/voltage condition
        self.w0 = w0 ## initial w-parameter condition
        self.v_thr = v_thr
        self.spike_reset = spike_reset

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals)
        self.v = Compartment(restVals + self.v0)
        self.w = Compartment(restVals + self.w0)
        self.s = Compartment(restVals)
        self.tols = Compartment(restVals) ## time-of-last-spike

    @compilable
    def advance_state(self, t, dt):
        j_mod = self.j.get() * self.resist_m
        if self.intgFlag == 1:
            v_params = (j_mod, self.w.get(), self.alpha, self.beta, self.gamma, self.tau_m)
            _, _v = step_rk2(0., self.v.get(), _dfv, dt, v_params)  # _v = step_rk2(v, v_params, _dfv, dt)
            w_params = (j_mod, self.v.get(), self.alpha, self.beta, self.gamma, self.tau_w)
            _, _w = step_rk2(0., self.w.get(), _dfw, dt, w_params)  # _w = step_rk2(w, w_params, _dfw, dt)
        else:  # integType == 0 (default -- Euler)
            v_params = (j_mod, self.w.get(), self.alpha, self.beta, self.gamma, self.tau_m)
            _, _v = step_euler(0., self.v.get(), _dfv, dt, v_params)  # _v = step_euler(v, v_params, _dfv, dt)
            w_params = (j_mod, self.v.get(), self.alpha, self.beta, self.gamma, self.tau_w)
            _, _w = step_euler(0., self.w.get(), _dfw, dt, w_params)  # _w = step_euler(w, w_params, _dfw, dt)
        s = (_v > self.v_thr) * 1.
        v = _v
        w = _w

        if self.spike_reset: ## if spike-reset used, variables snapped back to initial conditions
            v = v * (1. - s) + s * self.v0
            w = w * (1. - s) + s * self.w0

        ## update time-of-last spike variable(s)
        self.tols.set((1. - s) * self.tols.get() + (s * t))

        # self.j.set(j) ## j is not getting modified in these dynamics
        self.v.set(v)
        self.w.set(w)
        self.s.set(s)

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size, self.n_units))
        if not self.j.targeted:
            self.j.set(restVals)
        self.v.set(restVals + self.v0)
        self.w.set(restVals + self.w0)
        self.s.set(restVals)
        self.tols.set(restVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "FitzhughNagumoCell - evolves neurons according to nonlinear, "
                         "Fizhugh-Nagumo dual-ODE spiking cell dynamics."
        }
        compartment_props = {
            "inputs":
                {"j": "External input electrical current",
                 "key": "JAX PRNG key"},
            "states":
                {"v": "Membrane potential/voltage at time t",
                 "w": "Recovery variable at time t"},
            "outputs":
                {"s": "Emitted spikes/pulses at time t",
                 "tols": "Time-of-last-spike"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "tau_m": "Cell membrane time constant",
            "resist_m": "Membrane resistance value",
            "tau_w": "Recovery variable time constant",
            "v_thr": "Base voltage threshold value",
            "spike_reset": "Should voltage/recover be snapped to initial condition(s) if spike emitted?",
            "alpha": "Dimensionless recovery variable shift factor `a",
            "beta": "Dimensionless recovery variable scale factor `b`",
            "gamma": "Power-term divisor constant",
            "v0": "Initial condition for membrane potential/voltage",
            "w0": "Initial condition for recovery variable",
            "integration_type": "Type of numerical integration to use for the cell dynamics"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = v - (v^3)/3 - w + j * resist_m; "
                            "tau_w * dw/dt = v + a - b * w",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = FitzhughNagumoCell("X", 9)
    print(X)
