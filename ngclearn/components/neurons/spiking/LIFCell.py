from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
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

@partial(jit, static_argnums=[7,8,9,10,11])
def run_cell(dt, j, v, v_thr, v_theta, rfr, skey, tau_m, R_m, v_rest, v_reset, refract_T):
    """
    Runs leaky integrator neuronal dynamics

    Args:
        dt: integration time constant (milliseconds, or ms)

        j: electrical current value

        v: membrane potential (voltage, in milliVolts or mV) value (at t)

        v_thr: base voltage threshold value (in mV)

        v_theta: threshold shift (homeostatic) variable (at t)

        rfr: refractory variable vector (one per neuronal cell)

        skey: PRNG key which, if not None, will trigger a single-spike constraint
            (i.e., only one spike permitted to emit per single step of time);
            specifically used to randomly sample one of the possible action
            potentials to be an emitted spike

        tau_m: cell membrane time constant

        R_m: membrane resistance value

        v_rest: membrane resting potential (in mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value

        refract_T: (relative) refractory time period (in ms; Default
            value is 1 ms)

    Returns:
        voltage(t+dt), spikes, raw spikes, updated refactory variables
    """
    _v_thr = v_theta + v_thr ## calc present voltage threshold
    mask = (rfr >= refract_T).astype(jnp.float32) # get refractory mask
    ## update voltage / membrane potential
    _v = v + (v_rest - v) * (dt/tau_m) + (j * mask)
    ## obtain action potentials
    s = (_v > _v_thr).astype(jnp.float32)
    ## update refractory variables
    _rfr = (rfr + dt) * (1. - s)
    ## perform hyper-polarization of neuronal cells
    _v = _v * (1. - s) + s * v_reset

    raw_s = s + 0 ## preserve un-altered spikes
    ############################################################################
    ## this is a spike post-processing step
    if skey is not None: ## FIXME: this would not work for mini-batches!!!!!!!
        m_switch = (jnp.sum(s) > 0.).astype(jnp.float32)
        rS = random.choice(skey, s.shape[1], p=jnp.squeeze(s, axis=0))
        rS = nn.one_hot(rS, num_classes=s.shape[1], dtype=jnp.float32)
        s = s * (1. - m_switch) + rS * m_switch
    ############################################################################
    return _v, s, raw_s, _rfr

@partial(jit, static_argnums=[3,4])
def update_theta(dt, v_theta, s, tau_theta, theta_plus=0.05):
    """
    Runs homeostatic threshold update dynamics one step.

    Args:
        dt: integration time constant (milliseconds, or ms)

        v_theta: current value of homeostatic threshold variable

        s: current spikes (at t)

        tau_theta: homeostatic threshold time constant

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

    Returns:
        updated homeostatic threshold variable
    """
    #theta_decay = 0.9999999 #0.999999762 #jnp.exp(-dt/1e7)
    #theta_plus = 0.05
    #_V_theta = V_theta * theta_decay + S * theta_plus
    theta_decay = jnp.exp(-dt/tau_theta)
    _v_theta = v_theta * theta_decay + s * theta_plus
    #_V_theta = V_theta + -V_theta * (dt/tau_theta) + S * alpha
    return _v_theta

class LIFCell(Component): ## leaky integrate-and-fire cell
    """
    A spiking cell based on leaky integrate-and-fire (LIF) neuronal dynamics.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        R_m: membrane resistance value

        thr: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV)

        v_rest: membrane resting potential (in mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value

        tau_theta: homeostatic threshold time constant

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

        refract_T: relative refractory period time (ms; Default: 1 ms)

        one_spike: if True, a single-spike constraint will be enforced for
            every time step of neuronal dynamics simulated, i.e., at most, only
            a single spike will be permitted to emit per step -- this means that
            if > 1 spikes emitted, a single action potential will be randomly
            sampled from the non-zero spikes detected

        key: PRNG key to control determinism of any underlying random values
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save LIF parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, n_units, tau_m, R_m, thr=-52., v_rest=-65., v_reset=-60., # 60.
                 tau_theta=1e7, theta_plus=0.05, refract_T=5., key=None, one_spike=True,
                 useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.R_m = R_m ## resistance value
        self.one_spike = one_spike ## True => constrains system to simulate 1 spike per time step

        self.v_rest = v_rest #-65. # mV
        self.v_reset = v_reset # -60. # -65. # mV (milli-volts)
        self.tau_theta = tau_theta ## threshold time constant # ms (0 turns off)
        self.theta_plus = theta_plus #0.05 ## threshold increment
        self.refract_T = refract_T #5. # 2. ## refractory period  # ms

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

        # self.threshold = thr ## (fixed) base value for threshold  #-52 # -72. # mV
        # ## adaptive threshold setup
        # if directory is None:
        #     self.threshold_theta = jnp.zeros((1, n_units))
        # else:
        #     self.load(directory)

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals)
        self.v = Compartment(restVals + self.v_rest)
        self.s = Compartment(restVals)
        self.rfr = Compartment(restVals + self.refract_T)
        self.thr = Compartment(restVals + thr)
        self.thr_theta = Compartment(restVals)
        self.tols = Compartment(restVals) ## time-of-last-spike
        self.key = Compartment(random.PRNGKey(time.time_ns()) if key is None else key)
        self.clock_t = Compartment(0.) ## time-of-last-spike

        #self.reset()

    @staticmethod
    def pure_advance(t, dt, tau_m, R_m, v_rest, v_reset, refract_T, tau_theta,
                     theta_plus, one_spike, key, j, v, s, rfr, thr, thr_theta,
                     tols, clock_t):
        skey = None ## this is an empty dkey if single_spike mode turned off
        if one_spike == True: ## old code ~> if self.one_spike is False:
            key, *subkeys = random.split(key, 2)
            skey = subkeys[0]
        ## run one integration step for neuronal dynamics
        v, s, raw_spikes, rfr = run_cell(dt, j, v, thr, thr_theta, rfr, skey,
                                         tau_m, R_m, v_rest, v_reset, refract_T)
        if tau_theta > 0.:
            ## run one integration step for threshold dynamics
            thr_theta = update_theta(dt, thr_theta, raw_spikes, tau_theta, theta_plus)
        ## update tols
        tols = update_times(clock_t, s, tols) #update_times(t, s, tols)
        return j, v, s, rfr, thr, thr_theta, tols, key

    @resolver(pure_advance, output_compartments=['j', 'v', 's', 'rfr', 'thr',
        'thr_theta', 'tols', 'key'])
    def advance(self, vals):
        j, v, s, rfr, thr, thr_theta, tols, clock_t, key = vals
        self.j.set(j)
        self.v.set(v)
        self.s.set(s)
        self.rfr.set(rfr)
        self.thr.set(thr)
        self.thr_theta.set(thr_theta)
        self.tols.set(tols)
        self.key.set(key)

    @staticmethod
    def pure_reset(batch_size, n_units, v_rest, refract_T):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals #+ 0
        v = restVals + v_rest
        s = restVals #+ 0
        rfr = restVals + refract_T
        thr = restVals #+ 0
        thr_theta = restVals
        tols = restVals #+ 0
        clock_t = 0.
        return j, v, s, rfr, thr, thr_theta, tols, clock_t

    @resolver(pure_reset, output_compartments=['j', 'v', 's', 'rfr', 'thr',
        'thr_theta', 'tols', 'clock_t', 'key'])
    def reset(self, vals):
        j, v, s, rfr, thr, thr_theta, tols, clock_t = vals
        self.j.set(j)
        self.v.set(v)
        self.s.set(s)
        self.rfr.set(rfr)
        self.thr.set(thr)
        self.thr_theta.set(thr_theta)
        self.tols.set(tols)
        self.clock_t.set(clock_t)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, threshold_theta=self.threshold_theta)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.threshold_theta = data['threshold_theta']

    # def verify_connections(self):
    #     self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

# Testing
if __name__ == '__main__':
    from ngcsimlib.compartment import All_compartments
    from ngcsimlib.context import Context
    from ngcsimlib.commands import Command
    #from ngclearn.components.neurons.graded.rateCell import RateCell

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

    with Context("Context") as context:
        #b = BernoulliCell("b", 2)
        #a = RateCell("a2", 2, 0.01)
        #a.j << b.outputs
        a = LIFCell("a1", n_units=1, tau_m=100., R_m=1.)
        cmd = AdvanceCommand(components=[a], command_name="Advance")

    T = 16
    dt = 0.1
    compiled_cmd, arg_order = cmd.compile(loops=1, param_generator=lambda loop_id: [loop_id + 1, dt])
    wrapped_cmd = wrapper(compiled_cmd)

    t = 0.
    for i in range(T): # i is "t"
        a.clock_t.set(t)
        a.j.set(jnp.asarray([[1.0]]))
        wrapped_cmd()
        t = t + dt
        print(f"---[ Step {i} ]---")
        print(f"[a] j: {a.j.value}, v: {a.v.value}, s: {a.s.value}, " \
              f"rfr: {a.rfr.value}, thr: {a.thr.value}, theta: {a.thr_theta.value}, " \
              f"tols: {a.tols.value}, clock: {a.clock_t.value}")
    a.reset()
    print(f"---[ After reset ]---")
    print(f"[a] j: {a.j.value}, v: {a.v.value}, s: {a.s.value}, " \
          f"rfr: {a.rfr.value}, thr: {a.thr.value}, theta: {a.thr_theta.value}, " \
          f"tols: {a.tols.value}, clock: {a.clock_t.value}")
