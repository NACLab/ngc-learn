from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats

from ngclearn import numpy as jnp
import time, sys

class GatedTrace(Component): ## gated/piecewise low-pass filter
    """
    A gated/piecewise variable trace (filter).

    | --- Cell Input Compartments: (Takes wired-in signals) ---
    | inputs - input (takes wired-in external signals)
    | --- Cell Output Compartments: (These signals are generated) ---
    | trace - traced value signal

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant (ms)

        tau_tr: trace time constant (in milliseconds, or ms)
    """

    # Define Functions
    def __init__(self, name, n_units, dt, tau_tr, **kwargs):
        super().__init__(name, **kwargs)

        ## trace control coefficients
        self.dt = dt
        self.tau_tr = tau_tr ## trace time constant

        ## Layer size setup
        self.batch_size = 1
        self.n_units = n_units

        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input compartment
        self.trace = Compartment(restVals)

    @staticmethod
    def _advance_state(dt, tau_tr, inputs, trace):
        trace = (trace * (1. - dt/tau_tr)) * (1. - inputs) + inputs
        return trace

    @resolver(_advance_state)
    def advance_state(self, trace):
        self.trace.set(trace)

    @staticmethod
    def _reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        return restVals, restVals

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
