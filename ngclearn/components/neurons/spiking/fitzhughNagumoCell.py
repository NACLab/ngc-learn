from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
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

    # Define Functions
    def __init__(self, name, n_units, tau_m=1., tau_w=12.5, alpha=0.7,
                 beta=0.8, gamma=3., v_thr=1.07, v0=0., w0=0.,
                 integration_type="euler", key=None, useVerboseDict=False,
                 **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

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

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals)
        self.v = Compartment(restVals + self.v0)
        self.w = Compartment(restVals + self.w0)
        self.s = Compartment(restVals)
        self.tols = Compartment(restVals) ## time-of-last-spike

    @staticmethod
    def _advance_state(t, dt, tau_m, tau_w, v_thr, alpha, beta, gamma, intgFlag,
                       j, v, w, s, tols):
        v, w, s = run_cell(dt, j, v, w, v_thr, tau_m, tau_w, alpha, beta, gamma, intgFlag)
        tols = update_times(t, s, tols)
        return j, v, w, s, tols

    @resolver(_advance_state)
    def advance_state(self, j, v, w, s, tols):
        self.j.set(j)
        self.w.set(w)
        self.v.set(v)
        self.s.set(s)
        self.tols.set(tols)

    @staticmethod
    def _reset(batch_size, n_units, v0, w0):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals # None
        v = restVals + v0
        w = restVals + w0
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

## Testing
if __name__ == '__main__':
    from ngcsimlib.compartment import All_compartments
    from ngcsimlib.context import Context
    from ngcsimlib.commands import Command

    def wrapper(compiled_fn):
        def _wrapped(*args):
            # vals = jax.jit(compiled_fn)(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            vals = compiled_fn(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            for key, value in vals.items():
                All_compartments[str(key)].set(value)
            return vals
        return _wrapped

    class AdvanceCommand(Command):
        compile_key = "advance"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.gather()
                component.advance(t=t, dt=dt)

    class ResetCommand(Command):
        compile_key = "reset"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.reset(t=t, dt=dt)

    with Context("Context") as context:
        a = FitzhughNagumoCell("a1", n_units=1, tau_m=100.)
        advance_cmd = AdvanceCommand(components=[a], command_name="Advance")
        reset_cmd = ResetCommand(components=[a], command_name="Reset")

    T = 20 #16
    dt = 1. # 0.1

    compiled_advance_cmd, _ = advance_cmd.compile()
    wrapped_advance_cmd = wrapper(jit(compiled_advance_cmd))

    compiled_reset_cmd, _ = reset_cmd.compile()
    wrapped_reset_cmd = wrapper(jit(compiled_reset_cmd))

    t = 0.
    for i in range(T): # i is "t"
        a.j.set(jnp.asarray([[1.0]]))
        wrapped_advance_cmd(t, dt) ## pass in t and dt and run step forward of simulation
        t = t + dt
        print(f"---[ Step {i} ]---")
        print(f"[a] j: {a.j.value}, v: {a.v.value}, w: {a.w.value}, s: {a.s.value}, " \
              f"tols: {a.tols.value}")
    #a.reset()
    wrapped_reset_cmd()
    print(f"---[ After reset ]---")
    print(f"[a] j: {a.j.value}, v: {a.v.value}, w: {a.w.value}, s: {a.s.value}, " \
          f"tols: {a.tols.value}")
