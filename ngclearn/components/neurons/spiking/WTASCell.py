from jax import numpy as jnp, random, jit, nn
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit, nn
from functools import partial
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
from ngcsimlib.logger import info, warn

from ngcsimlib.compilers.process import transition
#from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngclearn.utils.model_utils import softmax


class WTASCell(JaxComponent): ## winner-take-all spiking cell
    """
    A spiking cell based on winner-take-all neuronal dynamics ("WTAS" stands
    for "winner-take-all-spiking").

    The differential equation for adjusting this specific cell
    (for adjusting v, given current j, over time) is:

    | tau_m * dv/dt = j * R  ;  v_p = softmax(v)
    | where R is membrane resistance and v_p is a voltage probability vector

    | --- Cell Input Compartments: ---
    | j - electrical current input (takes in external signals)
    | --- Cell State Compartments: ---
    | v - membrane potential/voltage state
    | rfr - (relative) refractory variable state
    | thr - (adaptive) threshold state
    | key - JAX PRNG key
    | --- Cell Output Compartments: ---
    | s - emitted binary spikes/action potentials
    | tols - time-of-last-spike

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        resist_m: membrane resistance value (Default: 1)

        thr_base: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV)

        thr_gain: increment to be applied to threshold in presence of spike

        refract_time: relative refractory period time (ms; Default: 1 ms)

        thr_jitter: scale of uniform jitter to add to initialization of thresholds
    """

    # Define Functions
    def __init__(
            self, name, n_units, tau_m, resist_m=1., thr_base=0.4, thr_gain=0.002, refract_time=0., thr_jitter=0.05,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        ## membrane parameter setup (affects ODE integration)
        self.tau_m = tau_m ## membrane time constant
        self.R_m = resist_m ## resistance value
        self.thr_gain = thr_gain
        self.thr_base = thr_base # mV ## base value for threshold
        self.refract_T = refract_time

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        ## base threshold setup
        ## according to eqn 26 of the source paper, the initial condition for the
        ## threshold should technically be between: 1/n_units < threshold0 << 0.5, e.g., 0.15
        key, subkey = random.split(self.key.value)
        self.threshold0 = thr_base + random.uniform(subkey, (1, n_units),
                                                   minval=-thr_jitter, maxval=thr_jitter,
                                                   dtype=jnp.float32)

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals)
        self.v = Compartment(restVals)
        self.s = Compartment(restVals)
        self.thr = Compartment(self.threshold0)
        self.rfr = Compartment(restVals + self.refract_T)
        self.tols = Compartment(restVals) ## time-of-last-spike

    @transition(output_compartments=["v", "s", "thr", "rfr", "tols"])
    @staticmethod
    def advance_state(t, dt, tau_m, R_m, thr_gain, refract_T, j, v, thr, rfr, tols):
        mask = (rfr >= refract_T) * 1.  ## check refractory period
        v = (j * R_m) * mask
        vp = softmax(v)  # convert to Categorical (spike) probabilities
        # s = nn.one_hot(jnp.argmax(vp, axis=1), j.shape[1]) ## hard-max spike
        s = (vp > thr) * 1. ## calculate action potential
        q = 1.  ## Note: thr_gain ==> "rho_b"
        ## increment threshold upon spike(s) occurrence
        dthr = jnp.sum(s, axis=1, keepdims=True) - q
        thr = jnp.maximum(thr + dthr * thr_gain, 0.025)  ## calc new threshold
        rfr = (rfr + dt) * (1. - s) + s * dt  # set refract to dt

        tols = (1. - s) * tols + (s * t) ## update tols
        return v, s, thr, rfr, tols

    @transition(output_compartments=["j", "v", "s", "rfr", "tols"])
    @staticmethod
    def reset(batch_size, n_units, refract_T):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals #+ 0
        v = restVals #+ 0
        s = restVals #+ 0
        rfr = restVals + refract_T
        tols = restVals #+ 0
        return j, v, s, rfr, tols

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, threshold=self.thr.value)

    def load(self, directory, seeded=False, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.thr.set( data['threshold'] )

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "WTASCell - evolves neurons according to winner-take-all "
                         "spiking dynamics "
        }
        compartment_props = {
            "inputs":
                {"j": "External input electrical current"},
            "states":
                {"v": "Membrane potential/voltage at time t",
                 "rfr": "Current state of (relative) refractory variable",
                 "thr": "Current state of voltage threshold at time t",
                 "key": "JAX PRNG key"},
            "outputs":
                {"s": "Emitted spikes/pulses at time t",
                 "tols": "Time-of-last-spike"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "tau_m": "Cell membrane time constant",
            "resist_m": "Membrane resistance value",
            "thr_base": "Base voltage threshold value",
            "thr_gain": "Amount to increment threshold by upon occurrence of spike",
            "refract_time": "Length of relative refractory period (ms)",
            "thr_jitter": "Scale of random uniform noise to apply to initial condition of threshold"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "tau_m * dv/dt = j * resist_m",
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
        X = WTASCell("X", 1, 1.)
    print(X)
