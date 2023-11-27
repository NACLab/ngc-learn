from ngclearn.engine.nodes.ops.op import Op
from jax import random, numpy as jnp, jit
from functools import partial

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

## Exponential spike kernel
class ExpKernel(Op):  # inherits from Node class
    """
    A spiking function based on an exponential kernel applied to
    a moving window of spike times.

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant

        tau_w: kernel time constant

        nu: spike time interval for window (window time bound)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this operator
    """
    def __init__(self, name, n_units, dt, tau_w=500., nu=4., key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        self.tau_w = tau_w  # kernel time constant; (micro-sec, spike window time constant)
        self.nu = nu  # window time bound (in ms?)
        self.win_len = int(nu/dt) + 1 # window length
        # cell compartments
        self.comp["in"] = None
        self.comp["epsp"] = None ## this emits a numeric "pulse" ()
        self.comp["tf"] = [] # window of spike times

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        s = self.comp["in"] ## get incoming spike readout
        tf = self.comp["tf"] ## get current window/volume

        _tf, epsp = apply_kernel(tf, s, self.t, self.tau_w, self.win_len,
                                 krn_start=0, krn_end=self.win_len-1) #0:win_len-1)
        self.comp["epsp"] = epsp ## get 2D batch matrix
        self.comp["tf"] = _tf ## update spike time window

    def set_to_rest(self, batch_size=1, hard=True):
        if hard:
            super().set_to_rest(batch_size)
            self.comp["tf"] = jnp.zeros([self.win_len, batch_size, self.n_units])

    def custom_dump(self, node_directory, template=False):
        required_keys = ['nu', 'tau_w']
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

    @staticmethod
    def get_default_out():
        """
        Returns the value within compartment ``epsp``
        """
        return 'epsp'

class_name = ExpKernel.__name__
