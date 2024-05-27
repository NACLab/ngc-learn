from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from ngclearn.utils import tensorstats

from jax import numpy as jnp
import time, sys

class LIFCell(Component): ## Lava-compliant leaky integrate-and-fire cell

    # Define Functions
    def __init__(self, name, n_units, thr_theta0, tau_m, R_m=1., thr=-52.,
                 v_rest=-65., v_reset=-60., v_decay=1., tau_theta=1e7, theta_plus=0.05,
                 refract_T=5., **kwargs):
        super().__init__(name, **kwargs)

        ## Cell dynamics setup
        self.tau_m = tau_m ## membrane time constant
        self.R_m = R_m ## resistance value
        self.v_rest = v_rest # mV
        self.v_reset = v_reset # mV (milli-volts)
        self.v_decay = v_decay
        self.tau_theta = tau_theta ## threshold time constant # ms (0 turns off)
        self.theta_plus = theta_plus ## threshold increment
        self.refract_T = refract_T ## refractory period  # ms
        self.thr = thr ## (fixed) base value for threshold # mV
        self.thr_theta0 = thr_theta0 ## initial jittered adaptive threshold values

        ## Layer size setup
        self.batch_size = 1
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals)
        self.v = Compartment(restVals + self.v_rest)
        self.s = Compartment(restVals)
        self.rfr = Compartment(restVals + self.refract_T)
        self.thr_theta = Compartment(thr_theta0)
        self.tols = Compartment(restVals) ## time-of-last-spike
        #self.reset()

    @staticmethod
    def _advance_state(t, dt, tau_m, R_m, v_rest, v_reset, v_decay, refract_T, thr, tau_theta,
                       theta_plus, j, v, s, rfr, thr_theta, tols):
        #j = j * (tau_m/dt) ## scale electrical current
        mask = (rfr >= refract_T) * 1. #numpy.greater_equal(rfr, refract_T) * 1.
        ## update voltage / membrane potential
        ### note: the ODE is a bit differently formulated here than usual
        dv_dt = (v_rest - v) * v_decay * (dt/tau_m) + ((j * R_m) * mask)
        v = v + dv_dt ### hard-coded Euler integration
        ## obtain action potentials/spikes
        s = (v > (thr + thr_theta)) * 1. #numpy.greater_equal(v, thr + thr_theta) * 1.
        ## update refractory variables
        rfr = (rfr + dt) * (1. - s)
        ## perform hyper-polarization of neuronal cells
        v = v * (1. - s) + s * v_reset
        ## update adaptive threshold variables
        theta_decay = jnp.exp(-dt/tau_theta)
        thr_theta = thr_theta * theta_decay + s * theta_plus
        ## update time-of-last-spike
        tols = (1. - s) * tols + (s * t)
        return v, s, rfr, thr_theta, tols

    @resolver(_advance_state)
    def advance_state(self, v, s, rfr, thr_theta, tols):
        self.v.set(v)
        self.s.set(s)
        self.rfr.set(rfr)
        self.thr_theta.set(thr_theta)
        self.tols.set(tols)

    @staticmethod
    def _reset(batch_size, n_units, v_rest, refract_T):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals #+ 0
        v = restVals + v_rest
        s = restVals #+ 0
        rfr = restVals + refract_T
        #thr_theta = thr_theta0 ## do not reset thr_theta
        tols = restVals #+ 0
        return j, v, s, rfr, tols

    @resolver(_reset)
    def reset(self, j, v, s, rfr, tols):
        self.j.set(j)
        self.v.set(v)
        self.s.set(s)
        self.rfr.set(rfr)
        #self.thr_theta.set(thr_theta)
        self.tols.set(tols)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name,
                  threshold_theta=self.thr_theta.value)

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.thr_theta.set( data['threshold_theta'] )

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
