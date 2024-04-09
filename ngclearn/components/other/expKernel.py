from ngcsimlib.component import Component
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

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'in'

    @classmethod
    def outputCompartmentName(cls):
        return 'out'

    ## Bind Properties to Compartments for ease of use
    @property
    def inputCompartment(self):
      return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, inp):
      self.compartments[self.inputCompartmentName()] = inp

    @property
    def epsp(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @epsp.setter
    def epsp(self, inp):
        self.compartments[self.outputCompartmentName()] = inp

    # Define Functions
    def __init__(self, name, n_units, tau_w=500., nu=4., key=None,
                 useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

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

        # cell compartments
        # self.comp["in"] = None
        # self.comp["epsp"] = None ## this emits a numeric "pulse" ()
        # self.comp["tf"] = [] # window of spike times

        ##Layer Size Setup
        self.n_units = n_units
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(),
                                                 min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        #self.t = self.t + self.dt
        self.gather()
        s = self.inputCompartment #self.comp["in"] ## get incoming spike readout
        tf = self.tf #self.comp["tf"] ## get current window/volume

        _tf, epsp = apply_kernel(tf, s, self.t, self.tau_w, self.win_len,
                                 krn_start=0, krn_end=self.win_len-1) #0:win_len-1)
        self.tf = _tf ## get 2D batch matrix
        self.epsp = epsp ## update spike time window
        # self.comp["epsp"] = epsp ## get 2D batch matrix
        # self.comp["tf"] = _tf ## update spike time window
        #self.inputCompartment = None

    def reset(self, **kwargs):
        self.tf = jnp.zeros([self.win_len, batch_size, self.n_units])
        #self.comp["tf"] = jnp.zeros([self.win_len, batch_size, self.n_units])
        self.inputCompartment = None
        self.outputCompartment = None

    def save(self, **kwargs):
        pass
