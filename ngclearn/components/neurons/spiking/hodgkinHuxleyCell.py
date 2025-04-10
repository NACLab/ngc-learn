from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit, nn
from functools import partial
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
from ngcsimlib.logger import info, warn
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2, step_rk4

from ngcsimlib.compilers.process import transition
#from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment


def _calc_biophysical_constants(v): ## computes H-H biophysical constants (which are functions of voltage v)
    alpha_n_of_v = .01 * ((10 - v) / (jnp.exp((10. - v) / 10.) - 1.))
    beta_n_of_v = .125 * jnp.exp(-v / 80.)
    alpha_m_of_v = .1 * ((25 - v) / (jnp.exp((25. - v) / 10.) - 1.))
    beta_m_of_v = 4. * jnp.exp(-v / 18.)
    alpha_h_of_v = .07 * jnp.exp(-v / 20.)
    beta_h_of_v = 1. / (jnp.exp((30 - v) / 10.) + 1.)
    return alpha_n_of_v, beta_n_of_v, alpha_m_of_v, beta_m_of_v, alpha_h_of_v, beta_h_of_v

def _dv_dt(t, v, j, m, n, h, tau_v, g_Na, g_K, g_L, v_Na, v_K, v_L): ## ODE for membrane potential/voltage
    ## C dv/dt = j - g_Na * m^3 * h * (v - v_Na) - g_K * n^4 * (v - v_K) - g_L * (v - v_L)
    term1 = g_Na * jnp.power(m, 3) * h * (v - v_Na)
    term2 = g_K * jnp.power(n, 4) * (v - v_K)
    term3 = g_L * (v - v_L)
    return (j - term1 - term2 - term3) * (1. / tau_v)

def dv_dt(t, v, params):
    j, m, n, h, tau_v, g_Na, g_K, g_L, v_Na, v_K, v_L = params
    return _dv_dt(t, v, j, m, n, h, tau_v, g_Na, g_K, g_L, v_Na, v_K, v_L)

def _dx_dt(t, x, alpha_x_of_v, beta_x_of_v): ## ODE for channel/gate
    ## dx/dt = alpha_x(v) * (1 - x) - beta_x(v) * x
    return alpha_x_of_v * (1 - x) - beta_x_of_v * x

def dx_dt(t, x, params):
    alpha_x_of_v, beta_x_of_v = params
    return _dx_dt(t, x, alpha_x_of_v, beta_x_of_v)

class HodgkinHuxleyCell(JaxComponent): ## Hodgkin-Huxley spiking cell
    """
    A spiking cell based the Hodgkin-Huxley (H-H) 1952 set of dynamics for describing the ionic mechanisms that underwrite
    the initiation and propagation of action potentials within a (giant) squid axon.

    The four differential equations for adjusting this specific cell
    (for adjusting v, given current j, over time) are:

    | tau_v dv/dt = j - g_Na * m^3 * h * (v - v_Na) - g_K * n^4 * (v - v_K) - g_L * (v - v_L)
    | dn/dt = alpha_n(v) * (1 - n) - beta_n(v) * n
    | dm/dt = alpha_m(v) * (1 - m) - beta_m(v) * m
    | dh/dt = alpha_h(v) * (1 - h) - beta_h(v) * h
    | where alpha_x(v) and beta_x(v) are functions that produce relevant biophysical constant values
    | depending on which gate/channel is being probed (i.e., x = n or m or h)

    | --- Cell Input Compartments: ---
    | j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | n - dimensionless probabilities for potassium channel subunit activation
    | m - dimensionless probabilities for sodium channel subunit activation
    | h - dimensionless probabilities for sodium channel subunit inactivation
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | tols - time-of-last-spike

    | References:
    | Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of membrane current and its application to
    | conduction and excitation in nerve." The Journal of physiology 117.4 (1952): 500.
    |
    | Kistler, Werner M., Wulfram Gerstner, and J. Leo van Hemmen. "Reduction of the Hodgkin-Huxley equations to a
    | single-variable threshold model." Neural computation 9.5 (1997): 1015-1045.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_v: membrane time constant (Default: 1 ms)

        resist_m: membrane resistance value

        v_Na: sodium reversal potential

        v_K: potassium reversal potential

        v_L: leak reversal potential

        g_Na: sodium (Na) conductance per unit area

        g_K: potassium (K) conductance per unit area

        g_L: leak conductance per unit area

        thr: voltage/membrane threshold (to obtain action potentials in terms of binary spikes/pulses)

        spike_reset: if True, once voltage crosses threshold, then dynamics of voltage and recovery are reset/snapped
            to `v_reset` which has a default value of 0 mV (Default: False)

        v_reset: voltage value to reset to after a spike (in mV)
            (Default: 0 mV)

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration), 
            "midpoint" or "rk2" (midpoint method/RK-2 integration), or 
            "rk4" (RK-4 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint or rk4 method will
                increase the accuracy of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)
    """

    # Define Functions
    def __init__(
            self, name, n_units, tau_v, resist_m=1., v_Na=115., v_K=-35., v_L=10.6, g_Na=100., g_K=5., g_L=0.3, thr=4.,
            spike_reset=False, v_reset=0., integration_type="euler", **kwargs
    ):
        super().__init__(name, **kwargs)

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## cell properties / biophysical parameter setup (affects ODE integration)
        self.tau_v = tau_v ## membrane time constant
        self.R_m = resist_m ## resistance value
        self.spike_reset = spike_reset
        self.thr = thr # mV ## base value for threshold
        self.v_reset = v_reset ## base value to reset voltage to (if spike_reset = True)
        self.v_Na = v_Na #115. ## ENa
        self.v_K = v_K #-35. #-12. ## EK
        self.v_L = v_L #10.6 ## EKleak
        self.g_Na = g_Na #100. #120. ## gNa
        self.g_K = g_K #5. #36. ## gK
        self.g_L = g_L #0.3 ## gKleak

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals, display_name="Electrical input current")
        self.v = Compartment(restVals, display_name="Membrane potential/voltage")
        self.n = Compartment(restVals, display_name="Potassium channel subunit activation (probability)")
        self.m = Compartment(restVals, display_name="Sodium channel subunit activation (probability)")
        self.h = Compartment(restVals, display_name="Sodium channel subunit inactivation (probability)")
        self.s = Compartment(restVals, display_name="Spike pulse")
        self.tols = Compartment(restVals, display_name="Time-of-last-spike") ## time-of-last-spike

    @transition(output_compartments=["v", "m", "n", "h", "s", "tols"])
    @staticmethod
    def advance_state(
            t, dt, spike_reset, v_reset, thr, tau_v, R_m, g_Na, g_K, g_L, v_Na, v_K, v_L, j, v, m, n, h, tols, intgFlag
    ):
        _j = j * R_m
        alpha_n_of_v, beta_n_of_v, alpha_m_of_v, beta_m_of_v, alpha_h_of_v, beta_h_of_v = _calc_biophysical_constants(v)
        ## integrate voltage / membrane potential
        if intgFlag == 1: ## midpoint method
            _, _v = step_rk2(0., v, dv_dt, dt, (_j, m + 0., n + 0., h + 0., tau_v, g_Na, g_K, g_L, v_Na, v_K, v_L))
            ## next, integrate different channels
            _, _n = step_rk2(0., n, dx_dt, dt, (alpha_n_of_v, beta_n_of_v))
            _, _m = step_rk2(0., m, dx_dt, dt, (alpha_m_of_v, beta_m_of_v))
            _, _h = step_rk2(0., h, dx_dt, dt, (alpha_h_of_v, beta_h_of_v))
        elif intgFlag == 4: ## Runge-Kutta 4th order
            _, _v = step_rk4(0., v, dv_dt, dt, (_j, m + 0., n + 0., h + 0., tau_v, g_Na, g_K, g_L, v_Na, v_K, v_L))
            ## next, integrate different channels
            _, _n = step_rk4(0., n, dx_dt, dt, (alpha_n_of_v, beta_n_of_v))
            _, _m = step_rk4(0., m, dx_dt, dt, (alpha_m_of_v, beta_m_of_v))
            _, _h = step_rk4(0., h, dx_dt, dt, (alpha_h_of_v, beta_h_of_v))
        else:  # integType == 0 (default -- Euler)
            _, _v = step_euler(0., v, dv_dt, dt, (_j, m + 0., n + 0., h + 0., tau_v, g_Na, g_K, g_L, v_Na, v_K, v_L))
            ## next, integrate different channels
            _, _n = step_euler(0., n, dx_dt, dt, (alpha_n_of_v, beta_n_of_v))
            _, _m = step_euler(0., m, dx_dt, dt, (alpha_m_of_v, beta_m_of_v))
            _, _h = step_euler(0., h, dx_dt, dt, (alpha_h_of_v, beta_h_of_v))
        ## obtain action potentials/spikes/pulses
        s = (_v > thr) * 1.
        if spike_reset:  ## if spike-reset used, variables snapped back to initial conditions
            alpha_n_of_v, beta_n_of_v, alpha_m_of_v, beta_m_of_v, alpha_h_of_v, beta_h_of_v = (
                _calc_biophysical_constants(v * 0 + v_reset))
            _v = _v * (1. - s) + s * v_reset
            _n = _n * (1. - s) + s * (alpha_n_of_v / (alpha_n_of_v + beta_n_of_v))
            _m = _m * (1. - s) + s * (alpha_m_of_v / (alpha_m_of_v + beta_m_of_v))
            _h = _h * (1. - s) + s * (alpha_h_of_v / (alpha_h_of_v + beta_h_of_v))
        ## transition to new state of (system of) variables
        v = _v
        m = _m
        n = _n
        h = _h
        tols = (1. - s) * tols + (s * t) ## update tols

        return v, m, n, h, s, tols

    @transition(output_compartments=["j", "v", "m", "n", "h", "s", "tols"])
    @staticmethod
    def reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        v = restVals  # + 0
        alpha_n_of_v, beta_n_of_v, alpha_m_of_v, beta_m_of_v, alpha_h_of_v, beta_h_of_v = _calc_biophysical_constants(v)
        j = restVals #+ 0
        n = alpha_n_of_v / (alpha_n_of_v + beta_n_of_v)
        m = alpha_m_of_v / (alpha_m_of_v + beta_m_of_v)
        h = alpha_h_of_v / (alpha_h_of_v + beta_h_of_v)
        s = restVals #+ 0
        tols = restVals #+ 0
        return j, v, m, n, h, s, tols

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        #jnp.savez(file_name, threshold=self.thr.value)

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        #self.thr.set( data['threshold'] )

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "WTASCell - evolves neurons according to winner-take-all "
                         "spiking dynamics "
        }
        compartment_props = {
            "inputs":
                {"j": "External input electrical current"},
            "states":
                {"v": "Membrane potential/voltage at time t",
                 "n": "Current state of potassium channel subunit activation",
                 "m": "Current state of sodium channel subunit activation",
                 "h": "Current state of sodium channel subunit inactivation",
                 "key": "JAX PRNG key"},
            "outputs":
                {"s": "Emitted spikes/pulses at time t",
                 "tols": "Time-of-last-spike"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "tau_v": "Cell membrane time constant",
            "resist_m": "Membrane resistance value",
            "thr": "Base voltage threshold value",
            "v_Na": "Sodium reversal potential",
            "v_K": "Potassium reversal potential",
            "v_L": "Leak reversal potential",
            "g_Na": "Sodium conductance per unit area",
            "g_K": "Potassium conductance per unit area",
            "g_L": "Leak conductance per unit area",
            "spike_reset": "Should this cell hyperpolarize by snapping to base values or not?",
            "v_reset": "Voltage value to reset to after a spike"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_v dv/dt = j - g_Na * m^3 * h * (v - v_Na) - g_K * n^4 * (v - v_K) - g_L * (v - v_L)",
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

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = HodgkinHuxleyCell("X", 1, 1.)
    print(X)
