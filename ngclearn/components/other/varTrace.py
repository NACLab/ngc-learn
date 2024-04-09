from ngcsimlib.component import Component
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

@partial(jit, static_argnums=[4])
def run_varfilter(dt, x, x_tr, decayFactor, a_delta=0.):
    """
    Run variable trace filter (low-pass filter) dynamics one step forward.

    Args:
        dt: integration time constant (ms)

        x: variable value / stimulus input (at t)

        x_tr: currenet value of trace/filter

        decayFactor: coefficient to decay trace by before application of new value

        a_delta: increment to made to filter (multiplied w/ stimulus x)
    Returns:
        updated trace/filter value/state
    """
    _x_tr = x_tr * decayFactor
    #x_tr + (-x_tr) * (dt / tau_tr) = (1 - dt/tau_tr) * x_tr
    if a_delta > 0.: ## perform additive form of trace ODE
        _x_tr = _x_tr + x * a_delta
        #_x_tr = x_tr + (-x_tr) * (dt / tau_tr) + x * a_delta
    else: ## run gated/piecewise ODE variant of trace
        _x_tr = _x_tr * (1. - x) + x
        #_x_tr = ( x_tr + (-x_tr) * (dt / tau_tr) ) * (1. - x) + x
    return _x_tr

class VarTrace(Component): ## low-pass filter
    """
    A variable trace (filter) functional node.

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        tau_tr: trace time constant (in milliseconds, or ms)

        a_delta: value to increment a trace by in presence of a spike; note if set
            to a value <= 0, then a piecewise gated trace will be used instead

        decay_type: string indicating the decay type to be applied to ODE
            integration; low-pass filter configuration

            :Note: string values that this can be (Default: "exp") are:
                1) `'lin'` = linear trace filter, i.e., decay = x_tr + (-x_tr) * (dt/tau_tr);
                2) `'exp'` = exponential trace filter, i.e., decay = exp(-dt/tau_tr) * x_tr;
                3) `'step'` = step trace, i.e., decay = 0 (a pulse applied upon input value)

        key: PRNG key to control determinism of any underlying random values
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save sLIF parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'in'

    @classmethod
    def outputCompartmentName(cls):
        return 'out'

    @classmethod
    def traceName(cls):
        return 'trace'

    ## Bind Properties to Compartments for ease of use
    @property
    def inputCompartment(self):
      return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, inp):
      self.compartments[self.inputCompartmentName()] = inp

    @property
    def trace(self):
        return self.compartments.get(self.traceName(), None)

    @trace.setter
    def trace(self, inp):
        self.compartments[self.traceName()] = inp

    # Define Functions
    def __init__(self, name, n_units, tau_tr, a_delta, decay_type="exp", key=None,
                 useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##TMP
        self.key, subkey = random.split(self.key)

        ## trace control coefficients
        self.tau_tr = tau_tr ## trace time constant
        self.a_delta = a_delta ## trace increment (if spike occurred)
        self.decay_type = decay_type ## lin --> linear decay; exp --> exponential decay
        self.decayFactor = None

        ##Layer Size Setup
        self.n_units = n_units
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(),
                                                 min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        if self.decayFactor is None: ## compute only once the decay factor
            self.decayFactor = 0. ## <-- pulse filter decay
            if "exp" in self.decay_type:
                self.decayFactor = jnp.exp(-dt/self.tau_tr)
            elif "lin" in self.decay_type:
                self.decayFactor = (1. - dt/self.tau_tr)
            ## else "step", yielding a step/pulse-like filter
        if self.trace is None:
            self.trace = jnp.zeros((1, self.n_units))
        s = self.inputCompartment
        self.trace = run_varfilter(dt, s, self.trace, self.decayFactor, self.a_delta)
        self.outputCompartment = self.trace
        #self.inputCompartment = None

    def reset(self, **kwargs):
        self.trace = jnp.zeros((1, self.n_units))
        self.outputCompartment = self.trace
        self.inputCompartment = None

    def save(self, **kwargs):
        pass
