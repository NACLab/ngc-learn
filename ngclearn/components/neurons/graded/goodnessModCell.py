from ngclib.component import Component
from jax import random, numpy as jnp, jit, nn
from jax import grad, value_and_grad #, jacfwd
from functools import partial
import time, sys

@partial(jit, static_argnums=[2])
def calc_goodness(z, thr, maximize=True):
    z_sqr = jnp.square(z) #tf.math.square(tf.matmul(z, self.R1))
    delta = jnp.sum(z_sqr, axis=1, keepdims=True)
    if maximize == True:
        ## maximize for positive samps, minimize for negative samps
        delta = delta - thr
    else:
        ## minimize for positive samps, maximize for negative samps
        delta = -delta + thr
    scale = 1. #5.
    delta = delta * scale #3.5
    # gets the probability P(pos)
    p = nn.sigmoid(delta)
    eps = 1e-5 #1e-6
    p = jnp.clip(p, eps, 1.0 - eps)
    return p, delta

@partial(jit, static_argnums=[3])
def calc_loss(z, lab, thr, keep_batch=False):
    _lab = (lab > 0.).astype(jnp.float32)
    p, logit = calc_goodness(z, thr)
    #CE = tf.nn.softplus(-logit) * lab + tf.nn.softplus(logit) * (1.0 - lab)
    CE = jnp.maximum(logit, 0) - logit * _lab + jnp.log(1. + jnp.exp(-jnp.abs(logit)))
    L = jnp.sum(CE, axis=1, keepdims=True)
    #CE = lab * tf.math.log(p) + (1.0 - lab) * tf.math.log(1.0 - p)
    #L = -tf.reduce_sum(CE, axis=1, keepdims=True)
    if keep_batch == False:
        L = jnp.mean(L) #jnp.sum(L)
    return L

@partial(jit, static_argnums=[3])
def calc_mod_signal(z, lab, thr, keep_batch):
    L, d_z = value_and_grad(calc_loss, argnums=0)(z, lab, thr, keep_batch)
    return L, d_z

class GoodnessModCell(Component): ## Contrastive real-valued modulator cell
    """
    The proposed contrastive / goodness modulator; this cell produces a
    signal based on a constrastive threshold-based functonal, i.e., "goodness",
    which produces a modulatory value equal to the first derivative of the
    contrastive functional.

    Args:
        name: the string name of this cell

        n_units:

        threshold: goodness threshold value (scalar)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'z'

    @classmethod
    def contrastLabelsName(cls):
        return 'lab'

    @classmethod
    def lossName(cls):
        return 'loss'

    @classmethod
    def modulatorName(cls):
        return 'modulator'

    ## Bind Properties to Compartments for ease of use

    @property
    def inputCompartment(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @inputCompartment.setter
    def inputCompartment(self, inp):
        self.compartments[self.inputCompartmentName()] = inp

    @property
    def contrastLabels(self):
        return self.compartments.get(self.contrastLabelsName(), None)

    @contrastLabels.setter
    def contrastLabels(self, inp):
        self.compartments[self.contrastLabelsName()] = inp

    @property
    def loss(self):
        return self.compartments.get(self.lossName(), None)

    @loss.setter
    def loss(self, inp):
        self.compartments[self.lossName()] = inp

    @property
    def modulator(self):
        return self.compartments.get(self.modulatorName(), None)

    @modulator.setter
    def modulator(self, inp):
        self.compartments[self.modulatorName()] = inp

    # Define Functions
    def __init__(self, name, n_units=1, threshold=7., key=None,
                 useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        self.goodness_threshold = threshold

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

        ## Set up bundle for multiple inputs of current
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)
        self.metadata.check_incoming_connections(self.contrastLabelsName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        keep_batch = False
        L, d_z = calc_mod_signal(self.inputCompartment, self.contrastLabels,
                                 self.goodness_threshold, keep_batch)
        self.loss = L
        self.modulator = d_z

    def reset(self, **kwargs):
        self.contrastLabels = None
        self.inputCompartment = None
        self.loss = None
        self.modulator = None

    def save(self, **kwargs):
        pass
