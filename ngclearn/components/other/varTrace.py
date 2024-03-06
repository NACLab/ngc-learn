from ngclib.component import Component
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

@partial(jit, static_argnums=[3,4,5])
def run_varfilter(dt, x, x_tr, tau_tr, a_delta=0., decay_type="lin"):
    """
    Variable trace filter (low-pass filter) dynamics
    """
    _x_tr = None
    if "exp" in decay_type: ## apply exponential decay
        gamma = jnp.exp(-dt/tau_tr)
        _x_tr = x_tr * gamma
    elif "lin" in decay_type: ## default => apply linear "lin" decay
        _x_tr = x_tr + (-x_tr) * (dt / tau_tr)
    elif "step" in decay_type:
        _x_tr = x_tr * 0
    else:
        print("ERROR: decay.type = {} unrecognized".format(decay_type))
        sys.exit(1)
    if a_delta > 0.: ## perform additive form of trace ODE
        _x_tr = _x_tr + x * a_delta
        #_x_tr = x_tr + (-x_tr) * (dt / tau_tr) + x * a_delta
    else: ## run gated/piecewise ODE variant of trace
        _x_tr = _x_tr * (1. - x) + x
        #_x_tr = ( x_tr + (-x_tr) * (dt / tau_tr) ) * (1. - x) + x
    return _x_tr


class VarTrace(Component): ## low-pass filter
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
      if inp is not None:
        if inp.shape[1] != self.n_units:
          raise RuntimeError("Input Compartment size does not match provided input size " + str(inp.shape) + "for "
                             + str(self.name))
      self.compartments[self.inputCompartmentName()] = inp

    @property
    def trace(self):
        return self.compartments.get(self.traceName(), None)

    @trace.setter
    def trace(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError(
                    "Input Compartment size does not match provided input size " + str(inp.shape) + "for "
                    + str(self.name))
        self.compartments[self.traceName()] = inp

    # Define Functions
    def __init__(self, name, n_units, tau_tr, a_delta, decay_type, key=None,
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

        ##Layer Size Setup
        self.n_units = n_units

        if directory is None:
            self.key, subkey = random.split(self.key)
        else:
            self.load(directory)

        ## Set up bundle for multiple inputs of current
        self.create_bundle('multi_input', 'additive')
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        if self.trace is None:
            self.trace = jnp.zeros((1, self.n_units))
        s = self.inputCompartment
        self.trace = run_varfilter(dt, s, self.trace, self.tau_tr, self.a_delta, decay_type=self.decay_type)
        self.outputCompartment = self.trace
        #self.inputCompartment = None

    def reset(self, **kwargs):
        self.trace = jnp.zeros((1, self.n_units))
        self.inputCompartment = None

    def save(self, **kwargs):
        pass
