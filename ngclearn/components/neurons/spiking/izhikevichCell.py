from ngcsimlib.component import Component
from jax import numpy as jnp, random, jit, nn
from functools import partial
import time, sys

@jit
def update_times(t, s, tols):
    """
    Updates time-of-last-spike (tols) variable.

    Args:
        t: current time (a scalar/int value)

        s: binary spike vector

        tols: current time-of-last-spike variable

    Returns:
        updated tols variable
    """
    _tols = (1. - s) * tols + (s * t)
    return _tols

@jit
def run_cell_euler(dt, j, v, s, w, thr=30., tau_m=1., tau_w=50., b=0.2, c=-65.,
                   d=8., R_m=1.):
    """
    Runs Izhikevich neuronal dynamics

    Args:
        dt: integration time constant (milliseconds, or ms)

        j: electrical current value

        v: membrane potential (voltage, in milliVolts or mV) value (at t)

        s: previously measured spikes/action potentials (binary values)

        w: recovery variable/state

        thr: voltage threshold value (in mV)

        tau_m: membrane time constant

        tau_w: (tau_recovery) time scale/constant of recovery variable; note
            that this is the inverse of Izhikevich's scale variable `a` (tau_w = 1/a)

        b: (coupling factor) how sensitive is recovery to subthreshold voltage
            fluctuations

        c: (reset_voltage) voltage to reset to after spike emitted (in mV)

        d: (reset_recovery) recovery value to reset to after a spike

        R_m: membrane resistance value (Default: 1 mega-Ohm)

    Returns:
        updated voltage, updated recovery, spikes
    """
    ## note: a = 0.1 --> fast spikes, a = 0.02 --> regular spikes
    a = 1./tau_w ## we map time constant to variable "a" (a = 1/tau_w)
    #_j = jnp.maximum(-30.0, j) ## lower-bound/clip input current
    ## check for spikes
    _s = (v > thr).astype(jnp.float32)
    ## for non-spikes, evolve according to dynamics
    _j = j * R_m
    dv = (jnp.square(v) * 0.04 + v * 5. + 140. - w + _j) * (1./tau_m)
    dw = (v * b - w) * a
    _v = v + dv * dt
    _w = w + dw * dt
    ## for spikes, snap to particular states
    _v = _v * (1. - _s) + _s * c
    _w = _w * (1. - _s) + _s * (w + d)
    return  _v, _w, _s

class IzhikevichCell(Component): ## Izhikevich neuronal cell
    """
    A spiking cell based on Izhikevich's model of neuronal dynamics. Note that
    this a two-variable simplification of more complex multi-variable systems
    (e.g., Hodgkin-Huxley model). This cell model iteratively evolves
    voltage "v" and recovery "w", the last of which represents the combined
    effects of sodium channel deinactivation and potassium channel deactivation.

    | References:
    | Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE Transactions
    | on neural networks 14.6 (2003): 1569-1572.

    Note: Izhikevich's constants/hyper-parameters 'a', 'b', 'c', and 'd' have
    been remapped/renamed (see arguments below). Note that the default settings
    for this component cell is for a regular spiking cell; to recover other
    types of spiking cells (depending on what neuronal circuitry one wants to
    model), one can recover specific models with the following particular values:

    | Intrinsically bursting neurons: ``v_reset=-55, w_reset=4``
    | Chattering neurons: ``v_reset = -50, w_reset=2``
    | Fast spiking neurons: ``tau_w = 10``
    | Low-threshold spiking neurons: ``coupling_factor = 0.25``
    | Resonator neurons: ``tau_w = 10, coupling_factor = 0.26``

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant (Default: 1 ms)

        R_m: membrane resistance value

        v_thr: voltage threshold value to cross for emitting a spike
            (in milliVolts, or mV) (Default: 30 mV)

        v_reset: voltage value to reset to after a spike (in mV)
            (Default: -65 mV), i.e., 'c'

        tau_w: recovery variable time constant (Default: 50 ms), i.e., 1/'a'

        w_reset: recovery value to reset to after a spike (Default: 8), i.e., 'd'

        coupling_factor: degree of to which recovery is sensitive to any
            subthreshold fluctuations of voltage (Default: 0.2), i.e., 'b'

        v0: initial condition / reset for voltage (Default: -65 mV)

        w0: initial condition / reset for recovery (Default: -14)

        key: PRNG key to control determinism of any underlying random values
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save Izhikevich parameter
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
    def voltageName(cls):
        return 'v'

    @classmethod
    def recoveryName(cls):
        return 'w'

    @classmethod
    def timeOfLastSpikeCompartmentName(cls):
        return 'tols'

    ## Bind Properties to Compartments for ease of use
    @property
    def inputCompartment(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, inp):
        self.compartments[self.inputCompartmentName()] = inp

    @property
    def outputCompartment(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @outputCompartment.setter
    def outputCompartment(self, out):
        self.compartments[self.outputCompartmentName()] = out

    @property
    def voltage(self):
        return self.compartments.get(self.voltageName(), None)

    @voltage.setter
    def voltage(self, t):
        self.compartments[self.voltageName()] = t

    @property
    def recovery(self):
        return self.compartments.get(self.recoveryName(), None)

    @recovery.setter
    def recovery(self, t):
        self.compartments[self.recoveryName()] = t

    @property
    def timeOfLastSpike(self):
        return self.compartments.get(self.timeOfLastSpikeCompartmentName(), None)

    @timeOfLastSpike.setter
    def timeOfLastSpike(self, t):
        self.compartments[self.timeOfLastSpikeCompartmentName()] = t

    # Define Functions
    def __init__(self, name, n_units, tau_m=1., R_m=1., v_thr=30., v_reset=-65.,
                 tau_w=50., w_reset=8., coupling_factor=0.2, v0=-65., w0=-14.,
                 key=None, useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ## Cell properties
        self.R_m = R_m
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.coupling = coupling_factor
        self.v_reset = v_reset
        self.w_reset = w_reset

        self.v0 = v0 ## initial membrane potential/voltage condition
        self.w0 = w0 ## initial recovery w-parameter condition
        self.v_thr = v_thr

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        j = self.inputCompartment
        v = self.voltage
        w = self.recovery
        s = self.outputCompartment
        #if self.integration_type == "euler":
        v, w, s = run_cell_euler(dt, j, v, s, w, thr=self.v_thr, tau_m=self.tau_m,
                                 tau_w=self.tau_w, b=self.coupling, c=self.v_reset,
                                 d=self.w_reset, R_m=self.R_m)
        self.voltage = v
        self.recovery = w
        self.outputCompartment = s

    def reset(self, **kwargs):
        self.inputCompartment = None
        self.voltage = jnp.zeros((self.batch_size, self.n_units)) + self.v0
        self.recovery = jnp.zeros((self.batch_size, self.n_units)) + self.w0
        self.outputCompartment = jnp.zeros((self.batch_size, self.n_units)) #None
        self.timeOfLastSpike = jnp.zeros((self.batch_size, self.n_units))

    def save(self, directory, **kwargs):
        pass
