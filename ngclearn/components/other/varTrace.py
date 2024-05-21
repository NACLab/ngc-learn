from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
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

    # Define Functions
    def __init__(self, name, n_units, tau_tr, a_delta, decay_type="exp", key=None,
                 useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ## trace control coefficients
        self.tau_tr = tau_tr ## trace time constant
        self.a_delta = a_delta ## trace increment (if spike occurred)
        self.decay_type = decay_type ## lin --> linear decay; exp --> exponential decay

        ##Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        self.inputs = Compartment(None) # input compartment
        self.outputs = Compartment(jnp.zeros((self.batch_size, self.n_units))) # output compartment
        self.trace = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        #self.reset()

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
        inputs = None
        return inputs, outputs, trace

    @resolver(_advance_state)
    def advance_state(self, inputs, outputs, trace):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.trace.set(trace)

    @staticmethod
    def _reset(batch_size, n_units):
        return None, jnp.zeros((batch_size, n_units)), jnp.zeros((batch_size, n_units))

    @resolver(_reset)
    def reset(self, inputs, outputs, trace):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.trace.set(trace)

## testing
if __name__ == '__main__':
    from ngcsimlib.compartment import All_compartments
    from ngcsimlib.context import Context
    from ngcsimlib.commands import Command
    from ngclearn.components.neurons.graded.rateCell import RateCell

    def wrapper(compiled_fn):
        def _wrapped(*args):
            # vals = jax.jit(compiled_fn)(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            vals = compiled_fn(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            for key, value in vals.items():
                All_compartments[str(key)].set(value)
            return vals
        return _wrapped

    class AdvanceCommand(Command):
        compile_key = "advance"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.gather()
                component.advance(t=t, dt=dt)

    class ResetCommand(Command):
        compile_key = "reset"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.reset(t=t, dt=dt)

    dkey = random.PRNGKey(1234)
    with Context("Bar") as bar:
        a = VarTrace("a", n_units=1, tau_tr=20., a_delta=0.02, decay_type="lin", key=dkey)
        advance_cmd = AdvanceCommand(components=[a], command_name="Advance")
        reset_cmd = ResetCommand(components=[a], command_name="Reset")

    compiled_advance_cmd, _ = advance_cmd.compile()
    wrapped_advance_cmd = wrapper(jit(compiled_advance_cmd))

    compiled_reset_cmd, _ = reset_cmd.compile()
    wrapped_reset_cmd = wrapper(jit(compiled_reset_cmd))

    T = 30
    dt = 1.

    t = 0. ## global clock
    for i in range(T):
        val = 0.
        if i % 5 == 0:
            val = 1.
        a.inputs.set(jnp.asarray([[val]]))
        wrapped_advance_cmd(t, dt)
        print(f"---[ Step {t} ]---")
        print(f"[a] inputs: {a.inputs.value}, outputs: {a.outputs.value}, trace: {a.trace.value}")
        t += dt
    wrapped_reset_cmd()
    print(f"---[ After reset ]---")
    print(f"[a] inputs: {a.inputs.value}, outputs: {a.outputs.value}, trace: {a.trace.value}")
