from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, nn, Array, jit
from ngcsimlib import deprecate_args
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2
from ngclearn.utils.surrogate_fx import (secant_lif_estimator, arctan_estimator,
                                         triangular_estimator,
                                         straight_through_estimator)

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment


@jit
def _dfv_internal(j, v, rfr, tau_m, refract_T): ## raw voltage dynamics
    mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    ## update voltage / membrane potential
    dv_dt = (j * mask) ## integration only involves electrical current
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, rfr, tau_m, refract_T = params
    dv_dt = _dfv_internal(j, v, rfr, tau_m, refract_T)
    return dv_dt

class IFCell(JaxComponent): ## integrate-and-fire cell
    """
    A spiking cell based on integrate-and-fire (IF) neuronal dynamics.

    The specific differential equation that characterizes this cell
    is (for adjusting v, given current j, over time) is:

    | tau_m * dv/dt = j * R
    | where R is the membrane resistance and v_rest is the resting potential
    | also, if a spike occurs, v is set to v_reset

    | --- Cell Input Compartments: ---
    | j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | rfr - (relative) refractory variable state
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | s_raw - raw spike signals before post-processing (only if one_spike = True, else s_raw = s)
    | tols - time-of-last-spike

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        resist_m: membrane resistance value (default: 1)

        thr: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV; default: -52. mV)

        v_rest: membrane resting potential (in mV; default: -65 mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value;
            (default: -60 mV)

        refract_time: relative refractory period time (ms; default: 0 ms)

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
                and "arctan" (arc-tangent estimator)

        lower_clamp_voltage: if True, this will ensure voltage never is below
            the value of `v_rest` (default: True)
    """

    @deprecate_args(thr_jitter=None)
    def __init__(
            self, name, n_units, tau_m, resist_m=1., thr=-52., v_rest=-65., v_reset=-60., refract_time=0.,
            integration_type="euler", surrogate_type="straight_through", lower_clamp_voltage=True, **kwargs
    ):
        super().__init__(name, **kwargs)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.resist_m = resist_m ## resistance value

        self.v_rest = v_rest #-65. # mV
        self.v_reset = v_reset # -60. # -65. # mV (milli-volts)
        ## basic asserts to prevent neuronal dynamics breaking...
        assert self.resist_m > 0.
        self.refract_T = refract_time #5. # 2. ## refractory period  # ms
        self.thr = thr ## (fixed) base value for threshold  #-52 # -72. # mV
        self.lower_clamp_voltage = lower_clamp_voltage

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        ## set up surrogate function for spike emission
        # if surrogate_type == "arctan":
        #     self.spike_fx, self.d_spike_fx = arctan_estimator()
        # elif surrogate_type == "triangular":
        #     self.spike_fx, self.d_spike_fx = triangular_estimator()
        # else: ## default: straight_through
        #     self.spike_fx, self.d_spike_fx = straight_through_estimator()


        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals, display_name="Current", units="mA")
        self.v = Compartment(restVals + self.v_rest,
                             display_name="Voltage", units="mV")
        self.s = Compartment(restVals, display_name="Spikes")
        self.rfr = Compartment(restVals + self.refract_T,
                               display_name="Refractory Time Period", units="ms")
        self.tols = Compartment(restVals, display_name="Time-of-Last-Spike",
                                units="ms") ## time-of-last-spike
        #self.surrogate = Compartment(restVals + 1., display_name="Surrogate State Value")

    @compilable
    def advance_state(
            self, dt, t
    ):
        ## run one integration step for neuronal dynamics
        j = self.j.get() * self.resist_m

        ### Runs integrator (or integrate-and-fire; IF) neuronal dynamics
        ## update voltage / membrane potential
        v_params = (j, self.rfr.get(), self.tau_m, self.refract_T)
        if self.intgFlag == 1:
            _, _v = step_rk2(0., self.v.get(), _dfv, dt, v_params)
        else:
            _, _v = step_euler(0., self.v.get(), _dfv, dt, v_params)
        ## obtain action potentials/spikes
        s = (_v > self.thr) * 1.
        ## update refractory variables
        rfr = (self.rfr.get() + dt) * (1. - s)
        ## perform hyper-polarization of neuronal cells
        v = _v * (1. - s) + s * self.v_reset

        #surrogate = d_spike_fx(v, self.thr)

        ## update tols
        self.tols.set((1. - s) * self.tols.get() + (s * t))
        if self.lower_clamp_voltage: ## ensure voltage never < v_rest
            _v = jnp.maximum(v, self.v_rest)

        self.v.set(_v)
        self.s.set(s)
        self.rfr.set(rfr)

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size, self.n_units))
        if not self.j.targeted:
            self.j.set(restVals)
        self.v.set(restVals + self.v_rest)
        self.s.set(restVals)
        self.rfr.set(restVals + self.refract_T)
        self.tols.set(restVals)
        #surrogate = restVals + 1.

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        ## constants loaded in
        self.tau_m = data['tau_m']
        self.thr = data['thr']
        self.v_rest = data['v_rest']
        self.v_reset = data['v_reset']
        self.v_decay = data['v_decay']
        self.resist_m = data['resist_m']
        self.tau_theta = data['tau_theta']
        self.theta_plus = data['theta_plus']

        if seeded:
            self.key.set(data['key'])

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "IFCell - evolves neurons according to integrate-"
                         "and-fire spiking dynamics."
        }
        compartment_props = {
            "inputs":
                {"j": "External input electrical current"},
            "states":
                {"v": "Membrane potential/voltage at time t",
                 "rfr": "Current state of (relative) refractory variable",
                 "thr": "Current state of voltage threshold at time t",
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
            "refract_time": "Length of relative refractory period (ms)",
            "integration_type": "Type of numerical integration to use for the cell dynamics",
            "surrgoate_type": "Type of surrogate function to use approximate "
                              "derivative of spike w.r.t. voltage/current",
            "lower_bound_clamp": "Should voltage be lower bounded to be never be below `v_rest`"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = (v_rest - v) + j * resist_m",
                "hyperparameters": hyperparams}
        return info

