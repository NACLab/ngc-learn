from jax import numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

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
def _dfv_internal(j, v, w, tau_m, omega, b): ## "voltage" dynamics
    # dy/dt =  omega x + b y
    dv_dt = omega * w + v * b ## dv/dt
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, w, tau_m, omega, b = params
    dv_dt = _dfv_internal(j, v, w, tau_m, omega, b)
    return dv_dt

@jit
def _dfw_internal(j, v, w, tau_w, omega, b): ## raw angular driver dynamics
    # dx/dt = b x − omega y + I; I is scaled injected electrical current
    dw_dt = w * b - v * omega + j
    dw_dt = dw_dt * (1./tau_w)
    return dw_dt

def _dfw(t, w, params): ## angular driver dynamics wrapper
    j, v, tau_w, omega, b = params
    dv_dt = _dfw_internal(j, v, w, tau_w, omega, b)
    return dv_dt

@jit
def _emit_spike(v, v_thr):
    s = (v > v_thr).astype(jnp.float32)
    return s

class RAFCell(JaxComponent):
    """
    The resonate-and-fire (RAF) neuronal cell
    model; a two-variable model. This cell model iteratively evolves
    voltage "v" and angular driver "w".

    The specific pair of differential equations that characterize this cell
    are (for adjusting v and w, given current j, over time):

    | tau_w * dw/dt = w * b - v * omega + j
    | tau_v * dv/dt = omega * w + v * b
    | where omega is angular frequency (Hz) and b is exponential dampening factor
    | Note: injected current j should generally be scaled by tau_w/dt

    | --- Cell Input Compartments: ---
    | j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | w - angular driver variable state
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | tols - time-of-last-spike

    | References:
    | Izhikevich, Eugene M. "Resonate-and-fire neurons." Neural networks
    | 14.6-7 (2001): 883-894.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_v: membrane/voltage time constant (Default: 1 ms)

        tau_w: angular driver variable time constant (Default: 1 ms)

        thr: voltage/membrane threshold (to obtain action potentials in terms
            of binary spikes) (Default: 1 mV)

        omega: angular frequency (Default: 10)

        b: oscillation dampening factor (Default: -1)

        v_reset: membrane potential reset condition (Default: 1 mV)

        w_reset: reset condition for angular current driver (Default: 0)

        v0: membrane potential initial condition (Default: 1 mV)

        w0: angular driver initial condition (Default: 0)

        resist_v: membrane resistance (Default: 1 mega-Ohm)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuray of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)
    """

    @deprecate_args(resist_m="resist_v", tau_m="tau_v")
    def __init__(self, name, n_units, tau_v=1., tau_w=1., thr=1., omega=10.,
                 b=-1., v_reset=1., w_reset=0., v0=0., w0=0., resist_v=1.,
                 integration_type="euler", batch_size=1, **kwargs):
        #v_rest=-72., v_reset=-75., w_reset=0., thr=5., v0=-70., w0=0., tau_w=400., thr=5., omega=10., b=-1.
        super().__init__(name, **kwargs)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Cell properties
        self.tau_v = tau_v
        self.resist_v = resist_v
        self.tau_w = tau_w
        self.omega = omega ## angular frequency
        self.b = b ## dampening factor
        ## note: the smaller b is, the faster the oscillation dampens to resting state values
        self.v_reset = v_reset
        self.w_reset = w_reset
        self.v0 = v0
        self.w0 = w0
        self.thr = thr

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals, display_name="Current", units="mA")
        self.v = Compartment(restVals + self.v0, display_name="Voltage", units="mV")
        self.w = Compartment(restVals + self.w0, display_name="Angular-Driver")
        self.s = Compartment(restVals, display_name="Spikes")
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike",
                                units="ms") ## time-of-last-spike

    @staticmethod
    def _advance_state(t, dt, tau_v, resist_v, tau_w, thr, omega, b,
                       v_reset, w_reset, intgFlag, j, v, w, tols):
        ## continue with centered dynamics
        j_ = j * resist_v
        if intgFlag == 1:  ## RK-2/midpoint
            w_params = (j_, v, tau_w, omega, b)
            _, _w = step_rk2(0., w, _dfw, dt, w_params)
            v_params = (j_, w, tau_v, omega, b)
            _, _v = step_rk2(0., v, _dfv, dt, v_params)
        else:  # integType == 0 (default -- Euler)
            w_params = (j_, v, tau_w, omega, b)
            _, _w = step_euler(0., w, _dfw, dt, w_params)
            v_params = (j_, w, tau_v, omega, b)
            _, _v = step_euler(0., v, _dfv, dt, v_params)
        s = _emit_spike(_v, thr)
        ## hyperpolarize/reset/snap variables
        w = _w * (1. - s) + s * w_reset
        v = _v * (1. - s) + s * v_reset

        tols = _update_times(t, s, tols)
        return j, v, w, s, tols

    @resolver(_advance_state)
    def advance_state(self, j, v, w, s, tols):
        self.j.set(j)
        self.w.set(w)
        self.v.set(v)
        self.s.set(s)
        self.tols.set(tols)

    @staticmethod
    def _reset(batch_size, n_units, v_reset, w_reset):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals # None
        v = restVals + v_reset
        w = restVals + w_reset
        s = restVals #+ 0
        tols = restVals #+ 0
        return j, v, w, s, tols

    @resolver(_reset)
    def reset(self, j, v, w, s, tols):
        self.j.set(j)
        self.v.set(v)
        self.w.set(w)
        self.s.set(s)
        self.tols.set(tols)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "RAFCell - evolves neurons according to nonlinear, "
                         "resonate-and-fire dual-ODE spiking cell dynamics."
        }
        compartment_props = {
            "inputs":
                {"j": "External input electrical current",
                 "key": "JAX PRNG key"},
            "states":
                {"v": "Membrane potential/voltage at time t",
                 "w": "Angular current driver variable at time t"},
            "outputs":
                {"s": "Emitted spikes/pulses at time t",
                 "tols": "Time-of-last-spike"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "tau_v": "Cell membrane time constant",
            "tau_w": "Recovery variable time constant",
            "v_reset": "Reset membrane potential value",
            "w_reset": "Reset angular driver value",
            "b": "Exponential dampening factor applied to oscillations",
            "omega": "Angular frequency of neuronal progress per second (radians)",
            "v0": "Initial condition for membrane potential/voltage",
            "w0": "Initial condition for membrane angular driver variable",
            "resist_v": "Membrane resistance value",
            "integration_type": "Type of numerical integration to use for the cell dynamics"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_v * dv/dt = omega * w + v * b; "
                            "tau_w * dw/dt = w * b - v * omega + j",
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
