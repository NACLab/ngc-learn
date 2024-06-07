from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats

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

class VarTrace(JaxComponent): ## low-pass filter
    """
    A variable trace (filter) functional node.

    | --- Cell Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signal (same as "trace" compartment)
    | trace - traced value signal

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
    """

    # Define Functions
    def __init__(self, name, n_units, tau_tr, a_delta, decay_type="exp", **kwargs):
        super().__init__(name, **kwargs)

        ## Trace control coefficients
        self.tau_tr = tau_tr ## trace time constant
        self.a_delta = a_delta ## trace increment (if spike occurred)
        self.decay_type = decay_type ## lin --> linear decay; exp --> exponential decay

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input compartment
        self.outputs = Compartment(restVals) # output compartment
        self.trace = Compartment(restVals)

    @staticmethod
    def _advance_state(t, dt, decay_type, tau_tr, a_delta, inputs, trace):
        ## compute the decay factor
        decayFactor = 0. ## <-- pulse filter decay (default)
        if "exp" in decay_type:
            decayFactor = jnp.exp(-dt/tau_tr)
        elif "lin" in decay_type:
            decayFactor = (1. - dt/tau_tr)
        ## else "step" == decay_type, yielding a step/pulse-like filter
        trace = run_varfilter(dt, inputs, trace, decayFactor, a_delta)
        outputs = trace
        return outputs, trace

    @resolver(_advance_state)
    def advance_state(self, outputs, trace):
        self.outputs.set(outputs)
        self.trace.set(trace)

    @staticmethod
    def _reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        return restVals, restVals, restVals

    @resolver(_reset)
    def reset(self, inputs, outputs, trace):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
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

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = VarTrace("X", 9, 0.0004, 3)
    print(X)
