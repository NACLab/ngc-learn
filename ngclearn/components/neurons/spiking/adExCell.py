from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit, nn
from ngcsimlib import deprecate_args
from ngcsimlib.logger import info, warn
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, step_euler, step_rk2

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

@jit
def _dfv_internal(j, v, w, tau_m, v_rest, sharpV, vT, R_m): ## raw voltage dynamics
    dv_dt = -(v - v_rest) + sharpV * jnp.exp((v - vT)/sharpV) - R_m * w + R_m * j ## dv/dt
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, w, tau_m, v_rest, sharpV, vT, R_m = params
    dv_dt = _dfv_internal(j, v, w, tau_m, v_rest, sharpV, vT, R_m)
    return dv_dt

@jit
def _dfw_internal(j, v, w, a, tau_w, v_rest): ## raw recovery dynamics
    dw_dt = -w + (v - v_rest) * a #+ b * s * tau_w
    dw_dt = dw_dt * (1./tau_w)
    return dw_dt

def _dfw(t, w, params): ## recovery dynamics wrapper
    j, v, a, tau_m, v_rest = params
    dv_dt = _dfw_internal(j, v, w, a, tau_m, v_rest)
    return dv_dt

class AdExCell(JaxComponent): ## adaptive exponential integrate-and-fire cell
    """
    The AdEx (adaptive exponential leaky integrate-and-fire) neuronal cell
    model; a two-variable model. This cell model iteratively evolves
    voltage "v" and recovery "w".

    The specific pair of differential equations that characterize this cell
    are (for adjusting v and w, given current j, over time):

    | tau_m * dv/dt = -(v - v_rest) + sharpV * exp((v - vT)/sharpV) - R_m * w + R_m * j
    | tau_w * dw/dt =  -w + (v - v_rest) * a
    | where w = w + s * (w + b) [in the event of a spike]

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
    | Brette, Romain, and Wulfram Gerstner. "Adaptive exponential integrate-and-fire
    | model as an effective description of neuronal activity." Journal of
    | neurophysiology 94.5 (2005): 3637-3642.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant (Default: 15 ms)

        resist_m: membrane resistance (Default: 1 mega-Ohm)

        tau_w: recover variable time constant (Default: 400 ms)

        v_sharpness: slope factor/sharpness constant (Default: 2)

        intrinsic_mem_thr: intrinsic membrane threshold (Default: -55 mV)

        thr: voltage/membrane threshold (to obtain action potentials in terms
            of binary spikes) (Default: 5 mV)

        v_rest: membrane resting potential (Default: -72 mV)

        a: adaptation coupling parameter (Default: 0.1)

        b: adaption/recover increment value (Default: 0.75)

        v0: initial condition / reset for voltage (Default: -70 mV)

        w0: initial condition / reset for recovery (Default: 0 mV)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuracy of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)
    """

    @deprecate_args(v_thr="thr")
    def __init__(
            self, name, n_units, tau_m=15., resist_m=1., tau_w=400., v_sharpness=2., intrinsic_mem_thr=-55., thr=5.,
            v_rest=-72., v_reset=-75., a=0.1, b=0.75, v0=-70., w0=0., integration_type="euler", batch_size=1, **kwargs
    ):
        super().__init__(name, **kwargs)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Cell properties
        self.tau_m = tau_m
        self.R_m = resist_m
        self.tau_w = tau_w
        self.sharpV = v_sharpness ## sharpness of action potential
        self.vT = intrinsic_mem_thr ## intrinsic membrane threshold
        self.a = a
        self.b = b
        self.v_rest = v_rest
        self.v_reset = v_reset

        self.v0 = v0 ## initial membrane potential/voltage condition
        self.w0 = w0 ## initial w-parameter condition
        self.thr = thr

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals, display_name="Current", units="mA")
        self.v = Compartment(restVals + self.v0, display_name="Voltage", units="mV")
        self.w = Compartment(restVals + self.w0, display_name="Recovery")
        self.s = Compartment(restVals, display_name="Spikes")
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike",
                                units="ms") ## time-of-last-spike

    @compilable
    def advance_state(self, t, dt):
        if self.intgFlag == 1:  ## RK-2/midpoint
            v_params = (self.j.get(), self.w.get(), self.tau_m, self.v_rest, self.sharpV, self.vT, self.R_m)
            _, _v = step_rk2(0., self.v.get(), _dfv, dt, v_params)
            w_params = (self.j.get(), self.v.get(), self.a, self.tau_w, self.v_rest)
            _, _w = step_rk2(0., self.w.get(), _dfw, dt, w_params)
        else:  # intgFlag == 0 (default -- Euler)
            v_params = (self.j.get(), self.w.get(), self.tau_m, self.v_rest, self.sharpV, self.vT, self.R_m)
            _, _v = step_euler(0., self.v.get(), _dfv, dt, v_params)
            w_params = (self.j.get(), self.v.get(), self.a, self.tau_w, self.v_rest)
            _, _w = step_euler(0., self.w.get(), _dfw, dt, w_params)
        s = (_v > self.thr) * 1. ## emit spikes/pulses
        ## hyperpolarize/reset/snap variables
        v = _v * (1. - s) + s * self.v_reset
        w = _w * (1. - s) + s * (_w + self.b)

        ## update time-of-last spike variable(s)
        self.tols.set((1. - s) * self.tols.get() + (s * t))

        #self.j.set(j) ## j is not getting modified in these dynamics
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
            "cell_type": "AdExCell - evolves neurons according to nonlinear, "
                         "adaptive exponential dual-ODE spiking cell dynamics."
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
            "batch_size": "Batch size dimension of this component",
            "tau_m": "Cell membrane time constant",
            "resist_m": "Membrane resistance value",
            "tau_w": "Recovery variable time constant",
            "thr": "Base voltage threshold value",
            "v_rest": "Resting membrane potential value",
            "v_reset": "Reset membrane potential value",
            "v_sharpness": "Slope factor/voltage sharpness constant",
            "intrinsic_mem_thr": "Intrinsic membrane threshold",
            "a": "Adaptation coupling parameter",
            "b": "Adaption/recover increment value",
            "v0": "Initial condition for membrane potential/voltage",
            "w0": "Initial condition for recovery variable",
            "integration_type": "Type of numerical integration to use for the cell dynamics"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = -(v - v_rest) + sharpV * exp((v - vT)/sharpV) - resist_m * w + resist_m * j; "
                            "tau_w * dw/dt =  -w + (v - v_rest) * a; where w = w + s * (w + b)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = AdExCell("X", 9)
    print(X)
