from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit, nn
from ngcsimlib import deprecate_args
from ngcsimlib.logger import info, warn
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, step_euler, step_rk2

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

@jit
def _dfv_internal(j, v, w, b, tau_m): ## raw voltage dynamics
    ## (v^2 * 0.04 + v * 5 + 140 - u + j) * a, where a = (1./tau_m) (w = u)
    dv_dt = (jnp.square(v) * 0.04 + v * 5. + 140. - w + j)
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, w, b, tau_m = params
    dv_dt = _dfv_internal(j, v, w, b, tau_m)
    return dv_dt

@jit
def _dfw_internal(j, v, w, b, tau_w): ## raw recovery dynamics
    ## (v * b - u) from (v * b - u) * a (Izh. form)  (w = u)
    dw_dt = (v * b - w)
    dw_dt = dw_dt * (1./tau_w)
    return dw_dt

def _dfw(t, w, params): ## recovery dynamics wrapper
    j, v, b, tau_w = params
    dv_dt = _dfw_internal(j, v, w, b, tau_w)
    return dv_dt

def _post_process(s, _v, _w, v, w, c, d): ## internal post-processing routine
    # this step is specific to izh neuronal cells, where, after dynamics
    # have evolved for a step in term, we then use the variables c and d
    # to gate accordingly the dynamics for voltage v and recovery w
    v_next = _v * (1. - s) + s * c
    w_next = _w * (1. - s) + s * (w + d)
    return v_next, w_next

class IzhikevichCell(JaxComponent): ## Izhikevich neuronal cell
    """
    A spiking cell based on Izhikevich's model of neuronal dynamics. Note that
    this a two-variable simplification of more complex multi-variable systems
    (e.g., Hodgkin-Huxley model). This cell model iteratively evolves
    voltage "v" and recovery "w", the last of which represents the combined
    effects of sodium channel deinactivation and potassium channel deactivation.

    The specific pair of differential equations that characterize this cell
    are (for adjusting v and w, given current j, over time):

    | tau_m * dv/dt = 0.04 v^2 + 5v + 140 - w + j * R_m
    | tau_w * dw/dt = (v * b - w),  where tau_w = 1/a

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
    | Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE Transactions
    | on neural networks 14.6 (2003): 1569-1572.

    Note: Izhikevich's constants/hyper-parameters 'a', 'b', 'c', and 'd' have
    been remapped/renamed (see arguments below). Note that the default settings
    for this component cell is for a regular spiking cell; to recover other
    types of spiking cells (depending on what neuronal circuitry one wants to
    model), one can recover specific models with the following particular values:

    | Intrinsically bursting neurons: ``v_reset=-55, w_reset=4``
    | Chattering neurons: ``v_reset = -50, w_reset = 2``
    | Fast spiking neurons: ``tau_w = 10``
    | Low-threshold spiking neurons: ``tau_w = 10, coupling_factor = 0.25, w_reset = 2``
    | Resonator neurons: ``tau_w = 10, coupling_factor = 0.26``

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant (Default: 1 ms)

        resist_m: membrane resistance value

        v_thr: voltage threshold value to cross for emitting a spike
            (in milliVolts, or mV) (Default: 30 mV)

        v_reset: voltage value to reset to after a spike (in mV)
            (Default: -65 mV), i.e., 'c'

        tau_w: recovery variable time constant (Default: 50 ms), i.e., 1/'a'

        w_reset: recovery value to reset to after a spike (Default: 8), i.e., 'd'

        coupling_factor: degree to which recovery is sensitive to any
            subthreshold fluctuations of voltage (Default: 0.2), i.e., 'b'

        v0: initial condition / reset for voltage (Default: -65 mV)

        w0: initial condition / reset for recovery (Default: -14)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuracy of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)
    """

    def __init__(self, name, n_units, tau_m=1., resist_m=1., v_thr=30., v_reset=-65.,
                 tau_w=50., w_reset=8., coupling_factor=0.2, v0=-65., w0=-14.,
                 integration_type="euler", **kwargs):
        super().__init__(name, **kwargs)

        ## Cell properties
        self.resist_m = resist_m ## resistance R_m
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.coupling_factor = coupling_factor
        self.v_reset = v_reset
        self.w_reset = w_reset

        self.v0 = v0 ## initial membrane potential/voltage condition
        self.w0 = w0 ## initial recovery w-parameter condition
        self.v_thr = v_thr

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

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
        ## note: a = 0.1 --> fast spikes, a = 0.02 --> regular spikes
        a = 1. / self.tau_w  ## we map time constant to variable "a" (a = 1/tau_w)
        _j = self.j.get() * self.resist_m
        # _j = jnp.maximum(-30.0, _j) ## lower-bound/clip input current
        ## check for spikes
        s = (self.v.get() > self.v_thr) * 1.
        ## for non-spikes, evolve according to dynamics
        if self.intgFlag == 1:
            v_params = (_j, self.w.get(), self.coupling_factor, self.tau_m)
            _, _v = step_rk2(0., self.v.get(), _dfv, dt, v_params)  # _v = step_rk2(v, v_params, _dfv, dt)
            w_params = (_j, self.v.get(), self.coupling_factor, self.tau_w)
            _, _w = step_rk2(0., self.w.get(), _dfw, dt, w_params)  # _w = step_rk2(w, w_params, _dfw, dt)
        else:  # integType == 0 (default -- Euler)
            v_params = (_j, self.w.get(), self.coupling_factor, self.tau_m)
            _, _v = step_euler(0., self.v.get(), _dfv, dt, v_params)  # _v = step_euler(v, v_params, _dfv, dt)
            w_params = (_j, self.v.get(), self.coupling_factor, self.tau_w)
            _, _w = step_euler(0., self.w.get(), _dfw, dt, w_params)  # _w = step_euler(w, w_params, _dfw, dt)
        ## for spikes, snap to particular states
        _v, _w = _post_process(s, _v, _w, self.v.get(), self.w.get(), self.v_reset, self.w_reset)
        v = _v
        w = _w

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
            "cell_type": "IzhikevichCell - evolves neurons according to nonlinear, "
                         "dual-ODE Izhikevich spiking cell dynamics."
        }
        compartment_props = {
            "inputs":
                {"j": "External input electrical current"},
            "states":
                {"v": "Membrane potential/voltage at time t",
                 "w": "Recovery variable at time t",
                 "key": "JAX PRNG key"},
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
            "v_rest": "Resting membrane potential value",
            "v_reset": "Reset membrane potential value",
            "w_reset": "Reset recover variable value",
            "coupling_factor": "Degree to which recovery variable is sensitive to subthreshold voltage fluctuations",
            "v0": "Initial condition for membrane potential/voltage",
            "w0": "Initial condition for recovery variable",
            "integration_type": "Type of numerical integration to use for the cell dynamics"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = 0.04 v^2 + 5v + 140 - w + j * resist_m; "
                            "tau_w * dw/dt = (v * b - w),  where tau_w = 1/a",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = IzhikevichCell("X", 9)
    print(X)
