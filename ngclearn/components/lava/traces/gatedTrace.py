from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn.utils import tensorstats
import time, sys


class GatedTrace(Component): ## gated/piecewise low-pass filter
    # Define Functions
    def __init__(self, name, n_units, tau_tr, **kwargs):
        super().__init__(name, **kwargs)

        ## trace control coefficients
        self.tau_tr = tau_tr ## trace time constant

        ## Layer size setup
        self.batch_size = 1
        self.n_units = n_units

        self.inputs = Compartment(None) # input compartment
        self.trace = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        #self.reset()

    @staticmethod
    def _advance_state(t, dt, tau_tr, inputs, trace):
        trace = (trace * (1. - dt/tau_tr)) * (1. - inputs) + inputs
        return trace

    @resolver(_advance_state)
    def advance_state(self, trace):
        self.trace.set(trace)

    @staticmethod
    def _reset(batch_size, n_units):
        return None, jnp.zeros((batch_size, n_units))

    @resolver(_reset)
    def reset(self, inputs, trace):
        self.inputs.set(inputs)
        self.trace.set(trace)

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
