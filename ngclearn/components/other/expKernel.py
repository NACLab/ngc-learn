from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from functools import partial
from ngclearn.utils import tensorstats
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

@partial(jit, static_argnums=[5,6])
def _apply_kernel(tf_curr, s, t, tau_w, win_len, krn_start, krn_end):
    idx_sub = tf_curr.shape[0]-1
    tf_new = (s * t) ## track new spike time(s)
    tf = tf_curr.at[idx_sub,:,:].set(tf_new)
    tf = jnp.roll(tf, shift=-1, axis=0) # 1,2,3  2,3,1

    ## apply exp time kernel
    ## EPSP = sum_{tf} exp(-(t - tf)/tau_w)
    _tf = tf[krn_start:krn_end,:,:]
    mask = jnp.greater(_tf, 0.).astype(jnp.float32) # 1 for every valid tf (non-zero)
    epsp = jnp.sum( jnp.exp( -(t - _tf)/tau_w ) * mask, axis=0 )
    return tf, epsp

class ExpKernel(JaxComponent): ## exponential kernel
    """
    A spiking function based on an exponential kernel applied to
    a moving window of spike times.

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell State Compartments: ---
    | tf - maintained local window of pulse signals
    | --- Cell Output Compartments: ---
    | epsp - excitatory postsynaptic potential/pulse

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant (the kernel needs access to this value)

        nu: (ms, spike time interval for window)

        tau_w: spike window time constant (in micro-secs, or nano-s)

        batch_size: batch size dimension of this cell (Default: 1)
    """

    def __init__(self, name, n_units, dt, tau_w=500., nu=4., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.tau_w = tau_w ## kernel window time constant
        self.nu = nu
        self.win_len = int(nu/dt) + 1 ## window length

        ## Layer Size Setup
        self.batch_size = batch_size
        self.n_units = n_units

        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.inputs = Compartment(restVals) # input comp
        self.epsp = Compartment(restVals) ## output comp
        ## window of spike times
        self.tf = Compartment(jnp.zeros((self.win_len, self.batch_size, self.n_units)))

    @compilable
    def advance_state(self, t):
        # Get the variables
        inputs = self.inputs.get()
        tf = self.tf.get()

        s = inputs
        ## update spike time window and corresponding window volume
        tf, epsp = _apply_kernel(
            tf, s, t, self.tau_w, self.win_len, krn_start=0, krn_end=self.win_len-1
        ) #0:win_len-1)

        # Update compartments
        self.epsp.set(epsp)
        self.tf.set(tf)

    @compilable
    def reset(self):
        restVals = jnp.zeros((self.batch_size, self.n_units)) ## inputs, epsp
        restTensor = jnp.zeros([self.win_len, self.batch_size, self.n_units], jnp.float32)  ## tf
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(restVals)
        self.epsp.set(restVals)
        self.tf.set(restTensor)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "ExpKernel - maintains an exponential kernel over "
                         "incoming signal values (such as sequences of discrete pulses)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"tr": "Value signal (rolling) time window"},
            "outputs":
                {"epsp": "Excitatory postsynaptic potential/pulse emitted at time t"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "dt": "Integration time constant (kernel needs knowledge of `dt`)",
            "nu": "Spike time interval for window",
            "tau_w": "Spike window time constant"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "epsp ~ Sum_{tf} exp(-(t - tf)/tau_w)",
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
        X = ExpKernel("X", 1, 1.)
    print(X)
