from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
from ngclearn import numpy as jnp
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info, warn

class LIFCell(Component): ## Lava-compliant leaky integrate-and-fire cell
    """
    A spiking cell based on (leaky) integrate-and-fire (LIF) neuronal dynamics.
    Note that this cell can be readily configured to pure integrate-and-fire
    dynamics as needed. Note that dynamics in this Lava-compliant cell are
    hard-coded to move according to Euler integration.

    The specific differential equation that characterize this cell
    is (for adjusting v, given current j, over time) is:

    | tau_m * dv/dt = gamma_d * (v_rest - v) + j * R
    | where R is the membrane resistance and v_rest is the resting potential
    | gamma_d is voltage decay -- 1 recovers LIF dynamics and 0 recovers IF dynamics

    | --- Cell Input Compartments: (Takes wired-in signals) ---
    | j_exc - excitatory electrical input
    | j_inh - inhibitory electrical input
    | --- Cell Output Compartments: (These signals are generated) ---
    | v - membrane potential/voltage state
    | s - emitted binary spikes/action potentials
    | rfr - (relative) refractory variable state
    | thr_theta - homeostatic/adaptive threshold increment state

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant (ms)

        tau_m: cell membrane time constant

        thr_theta_init: initialization kernel for threshold increment variable

        resist_m: membrane resistance value (Default: 1)

        thr: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV)

        v_rest: membrane resting potential (in mV)

        v_reset: membrane reset potential (in mV) -- upon occurrence of a spike,
            a neuronal cell's membrane potential will be set to this value

        v_decay: decay factor applied to voltage leak (Default: 1.); setting this
            to 0 mV results in pure integrate-and-fire (IF) dynamics

        tau_theta: homeostatic threshold time constant

        theta_plus: physical increment to be applied to any threshold value if
            a spike was emitted

        refract_time: relative refractory period time (ms; Default: 1 ms)

        thr_theta0: (DEPRECATED) initial conditions for voltage threshold
    """

    # Define Functions
    def __init__(self, name, n_units, dt, tau_m, thr_theta_init=None, resist_m=1.,
                 thr=-52., v_rest=-65., v_reset=-60., v_decay=1., tau_theta=1e7,
                 theta_plus=0.05, refract_time=5., thr_theta0=None, **kwargs):
        super().__init__(name, **kwargs)

        ## Cell dynamics setup
        self.dt = dt
        self.tau_m = tau_m ## membrane time constant
        self.R_m = resist_m ## resistance value
        if kwargs.get("R_m") is not None:
            warn("The argument `R_m` being used is deprecated.")
            self.Rscale = kwargs.get("R_m")
        self.v_rest = v_rest # mV
        self.v_reset = v_reset # mV (milli-volts)
        self.v_decay = v_decay
        ## basic asserts to prevent neuronal dynamics breaking...
        assert (self.v_decay * self.dt / self.tau_m) <= 1.
        assert self.R_m > 0.
        self.tau_theta = tau_theta ## threshold time constant # ms (0 turns off)
        self.theta_plus = theta_plus ## threshold increment
        self.refract_T = refract_time ## refractory period  # ms
        self.thr = thr ## (fixed) base value for threshold # mV
        self.thr_theta_init = thr_theta_init
        self.thr_theta0 = thr_theta0 ## initial jittered adaptive threshold values

        ## Component size setup
        self.batch_size = 1
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j_exc = Compartment(restVals)
        self.j_inh = Compartment(restVals)
        self.v = Compartment(restVals + self.v_rest)
        self.s = Compartment(restVals)
        self.rfr = Compartment(restVals + self.refract_T)
        self.thr_theta = Compartment(None)

        if thr_theta0 is not None:
            warn("The argument `thr_theta0` being used is deprecated.")
            self._init(thr_theta0)
        else:
            if self.thr_theta_init is None:
                info(self.name, "is using default threshold variable initializer!")
                self.thr_theta_init = {"dist": "constant", "value": 0.}
            thr_theta0 = initialize_params(None, self.thr_theta_init, (1, self.n_units))
            self._init(thr_theta0)

    def _init(self, thr_theta0):
        self.thr_theta.set(thr_theta0)

    @staticmethod
    def _advance_state(dt, tau_m, R_m, v_rest, v_reset, v_decay, refract_T, thr, tau_theta,
                       theta_plus, j_exc, j_inh, v, s, rfr, thr_theta):
        #j = j * (tau_m/dt) ## scale electrical current
        j = j_exc - j_inh ## sum the excitatory and inhibitory input channels
        mask = (rfr >= refract_T) * 1. #numpy.greater_equal(rfr, refract_T) * 1.
        ## update voltage / membrane potential
        ### note: the ODE is a bit differently formulated here than usual
        dv_dt = (v_rest - v) * v_decay * (dt/tau_m) + ((j * R_m) * mask)
        v = v + dv_dt ### hard-coded Euler integration
        ## obtain action potentials/spikes
        s = (v > (thr + thr_theta)) * 1. #numpy.greater_equal(v, thr + thr_theta) * 1.
        ## update refractory variables
        rfr = (rfr + dt) * (1. - s)
        ## perform hyper-polarization of neuronal cells
        v = v * (1. - s) + s * v_reset
        ## update adaptive threshold variables
        theta_decay = jnp.exp(-dt/tau_theta)
        thr_theta = thr_theta * theta_decay + s * theta_plus
        ## update time-of-last-spike
        #tols = (1. - s) * tols + (s * t)
        return v, s, rfr, thr_theta #, tols

    @resolver(_advance_state)
    def advance_state(self, v, s, rfr, thr_theta): #, tols):
        self.v.set(v)
        self.s.set(s)
        self.rfr.set(rfr)
        self.thr_theta.set(thr_theta)
        #self.tols.set(tols)

    @staticmethod
    def _reset(batch_size, n_units, v_rest, refract_T):
        restVals = jnp.zeros((batch_size, n_units))
        j_exc = restVals #+ 0
        j_inh = restVals #+ 0
        v = restVals + v_rest
        s = restVals #+ 0
        rfr = restVals + refract_T
        return j_exc, j_inh, v, s, rfr #, tols

    @resolver(_reset)
    def reset(self, j_exc, j_inh, v, s, rfr):#, tols):
        self.j_exc.set(j_exc)
        self.j_inh.set(j_inh)
        self.v.set(v)
        self.s.set(s)
        self.rfr.set(rfr)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name,
                  threshold_theta=self.thr_theta.value)

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self._init( data['threshold_theta'] )


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
