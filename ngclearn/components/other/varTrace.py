from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
from ngcsimlib.logger import info, warn

from ngcsimlib.compilers.process import transition
#from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

@partial(jit, static_argnums=[4])
def _run_varfilter(dt, x, x_tr, decayFactor, gamma_tr, a_delta=0.):
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
    _x_tr = gamma_tr * x_tr * decayFactor
    #x_tr + (-x_tr) * (dt / tau_tr) = (1 - dt/tau_tr) * x_tr
    if a_delta > 0.: ## perform additive form of trace ODE
        _x_tr = _x_tr + x * a_delta
        #_x_tr = x_tr + (-x_tr) * (dt / tau_tr) + x * a_delta
    else: ## run gated/piecewise ODE variant of trace
        _x_tr = _x_tr * (1. - x) + x
        #_x_tr = ( x_tr + (-x_tr) * (dt / tau_tr) ) * (1. - x) + x
    return _x_tr

class VarTrace(JaxComponent): ## low-pass filter
    """
    A variable trace (filter) functional node.

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell State Compartments: ---
    | trace - traced value signal
    | --- Cell Output Compartments: ---
    | outputs - output signal (same as "trace" compartment)
    | trace - traced value signal (can be treated as output compartment)

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        tau_tr: trace time constant (in milliseconds, or ms)

        a_delta: value to increment a trace by in presence of a spike; note if set
            to a value <= 0, then a piecewise gated trace will be used instead

        gamma_tr: an extra multiplier in front of the leak of the trace (Default: 1)

        decay_type: string indicating the decay type to be applied to ODE
            integration; low-pass filter configuration

            :Note: string values that this can be (Default: "exp") are:
                1) `'lin'` = linear trace filter, i.e., decay = x_tr + (-x_tr) * (dt/tau_tr);
                2) `'exp'` = exponential trace filter, i.e., decay = exp(-dt/tau_tr) * x_tr;
                3) `'step'` = step trace, i.e., decay = 0 (a pulse applied upon input value)

        batch_size: batch size dimension of this cell (Default: 1)
    """

    # Define Functions
    def __init__(self, name, n_units, tau_tr, a_delta, gamma_tr=1, decay_type="exp",
                 batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        ## Trace control coefficients
        self.tau_tr = tau_tr ## trace time constant
        self.a_delta = a_delta ## trace increment (if spike occurred)
        self.gamma_tr = gamma_tr
        self.decay_type = decay_type ## lin --> linear decay; exp --> exponential decay

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input compartment
        self.outputs = Compartment(restVals) # output compartment
        self.trace = Compartment(restVals)

    @transition(output_compartments=["outputs", "trace"])
    @staticmethod
    def advance_state(dt, decay_type, tau_tr, a_delta, gamma_tr, inputs, trace):
        decayFactor = 0.
        if "exp" in decay_type:
            decayFactor = jnp.exp(-dt/tau_tr)
        elif "lin" in decay_type:
            decayFactor = (1. - dt/tau_tr)
        _x_tr = gamma_tr * trace * decayFactor
        if a_delta > 0.:
            _x_tr = _x_tr + inputs * a_delta
        else:
            _x_tr = _x_tr * (1. - inputs) + inputs
        trace = _x_tr
        return trace, trace

    @transition(output_compartments=["inputs", "outputs", "trace"])
    @staticmethod
    def reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        return restVals, restVals, restVals

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "VarTrace - maintains a low pass filter over incoming signal "
                         "values (such as sequences of discrete pulses)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"trace": "Continuous low-pass filtered signal values, at time t"},
            "outputs":
                {"outputs": "Continuous low-pass filtered signal values, "
                            "at time t (same as `trace`)"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "tau_tr": "Trace/filter time constant",
            "a_delta": "Increment to apply to trace (if not set to 0); "
                       "otherwise, traces clamp to 1 and then decay",
            "decay_type": "Indicator of what type of decay dynamics to use "
                          "as filter is updated at time t"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_tr * dz/dt ~ -z + inputs",
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
        X = VarTrace("X", 9, 0.0004, 3)
    print(X)
