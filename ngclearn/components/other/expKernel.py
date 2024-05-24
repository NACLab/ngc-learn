from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

@partial(jit, static_argnums=[5,6])
def apply_kernel(tf_curr, s, t, tau_w, win_len, krn_start, krn_end):
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

class ExpKernel(Component): ## Exponential spike kernel
    """
    A spiking function based on an exponential kernel applied to
    a moving window of spike times.

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        nu: (ms, spike time interval for window)

        tau_w: spike window time constant (in micro-secs, or nano-s)

        key: PRNG key to control determinism of any underlying random values
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save sLIF parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, n_units, tau_w=500., nu=4., key=None,
                 directory=None, **kwargs):
        super().__init__(name, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##TMP
        self.key, subkey = random.split(self.key)

        ## trace control coefficients
        self.tau_w = tau_w ## kernel window time constant
        self.nu = nu
        self.win_len = int(nu/dt) + 1 ## window length
        self.tf = None #[] ## window of spike times

        ##Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        # cell compartments
        self.inputs = Compartment(None) # input compartment
        self.outputs = Compartment(jnp.zeros((self.batch_size, self.n_units))) # output compartment
        self.epsp = Compartment(jnp.zeros((self.batch_size, self.n_units)))
        #self.reset()

    @staticmethod
    def _advance_state(t, dt, decay_type, tau_w, win_len, inputs, epsp, tf):
        #self.t = self.t + self.dt
        #self.gather()
        ## get incoming spike readout and current window/volume
        s = inputs ## spike readout
        #tf = self.tf ## current window/volume
        ## update spike time window and corresponding window volume
        tf, epsp = apply_kernel(tf, s, t, tau_w, win_len, krn_start=0,
                                krn_end=win_len-1) #0:win_len-1)
        #self.tf = _tf ## get the corresponding 2D batch matrix
        #self.epsp = epsp ## update spike time window
        # self.comp["epsp"] = epsp ## get 2D batch matrix
        # self.comp["tf"] = _tf ## update spike time window
        #self.inputCompartment = None
        return epsp, tf

    @resolver(_advance_state)
    def advance_state(self, epsp, tf):
        self.epsp.set(epsp)
        self.tf.set(tf)

    @staticmethod
    def _reset(batch_size, n_units, win_len):
        #self.tf = jnp.zeros([self.win_len, batch_size, self.n_units])
        tf = jnp.zeros([win_len, batch_size, n_units])
        return None, jnp.zeros((batch_size, n_units)), tf

    @resolver(_reset)
    def reset(self, inputs, outputs, tf):
        self.inputs.set(inputs)
        self.epsp.set(epsp)
        self.tf.set(tf)
