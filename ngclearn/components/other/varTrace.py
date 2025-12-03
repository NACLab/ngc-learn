# %%

from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

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

        P_scale: if `a_delta=0`, then this scales the value that the trace snaps to upon receiving a pulse value

        gamma_tr: an extra multiplier in front of the leak of the trace (Default: 1)

        decay_type: string indicating the decay type to be applied to ODE
            integration; low-pass filter configuration

            :Note: string values that this can be (Default: "exp") are:
                1) `'lin'` = linear trace filter, i.e., decay = x_tr + (-x_tr) * (dt/tau_tr);
                2) `'exp'` = exponential trace filter, i.e., decay = exp(-dt/tau_tr) * x_tr;
                3) `'step'` = step trace, i.e., decay = 0 (a pulse applied upon input value)

        n_nearest_spikes: (k) if k > 0, this makes the trace act like a nearest-neighbor trace, 
            i.e., k = 1 yields the 1-nearest (neighbor) trace (Default: 0)

        batch_size: batch size dimension of this cell (Default: 1)
    """

    def __init__(self, name, n_units, tau_tr, a_delta, P_scale=1., gamma_tr=1, decay_type="exp",
                 n_nearest_spikes=0, batch_size=1, key=None):
        super().__init__(name, key)

        ## Trace control coefficients
        self.decay_type = decay_type ## lin --> linear decay; exp --> exponential decay

        self.tau_tr = tau_tr ## trace time constant
        self.a_delta = a_delta ## trace increment (if spike occurred)
        self.P_scale = P_scale ## trace scale if non-additive trace to be used
        self.gamma_tr = gamma_tr
        self.n_nearest_spikes = n_nearest_spikes

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input compartment
        self.outputs = Compartment(restVals) # output compartment
        self.trace = Compartment(restVals)

    @compilable
    def advance_state(self, dt):
        if "exp" in self.decay_type:
            decayFactor = jnp.exp(-dt/self.tau_tr)
        elif "lin" in self.decay_type:
            decayFactor = (1. - dt/self.tau_tr)
        else:
            decayFactor = 0.


        _x_tr = self.gamma_tr * self.trace.get() * decayFactor
        if self.n_nearest_spikes > 0:
            _x_tr = _x_tr + self.inputs.get() * (self.a_delta - (self.trace.get() / self.n_nearest_spikes))
        else:
            if self.a_delta > 0.:
                _x_tr = _x_tr + self.inputs.get() * self.a_delta
            else:
                _x_tr = _x_tr * (1. - self.inputs.get()) + self.inputs.get() * self.P_scale

        self.trace.set(_x_tr)
        self.outputs.set(_x_tr)


    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size, self.n_units))
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(restVals)
        self.outputs.set(restVals)
        self.trace.set(restVals)

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
            "P_scale": "Max value to snap trace to if a max-clamp trace is triggered/configured", 
            "decay_type": "Indicator of what type of decay dynamics to use "
                          "as filter is updated at time t", 
            "n_nearest_neighbors": "Number of nearest pulses to affect/increment trace (if > 0)"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_tr * dz/dt ~ -z + inputs * a_delta (full convolution trace); " 
                            "tau_tr * dz/dt ~ -z + inputs * (a_delta - z/n_nearest_neighbors) (near trace)",
                "hyperparameters": hyperparams}
        return info


if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = VarTrace("X", 9, 0.0004, 3)
    print(X)
