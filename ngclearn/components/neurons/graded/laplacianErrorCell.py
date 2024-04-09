from ngcsimlib.component import Component
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

#@partial(jit, static_argnums=[3])
def run_cell(dt, targ, mu, eType="gaussian"):
    """
    Moves cell dynamics one step forward.

    Args:
        dt: integration time constant

        targ: target pattern value

        mu: prediction value

    Returns:
        derivative w.r.t. mean "dmu", derivative w.r.t. target dtarg, local loss
    """
    return run_laplacian_cell(dt, targ, mu)

@jit
def run_laplacian_cell(dt, targ, mu):
    """
    Moves Laplacian cell dynamics one step forward. Specifically, this
    routine emulates the error unit behavior of the local cost functional:

    | L(targ, mu) = -||targ - mu||_1
    | or log likelihood of the Laplace distribution with identity scale

    Args:
        dt: integration time constant

        targ: target pattern value

        mu: prediction value

    Returns:
        derivative w.r.t. mean "dmu", derivative w.r.t. target dtarg, loss
    """
    dmu = jnp.sign(targ - mu) # e (error unit)
    dtarg = -dmu # reverse of e
    L = -jnp.sum(jnp.abs(dmu)) # technically, this is mean absolute error
    return dmu, dtarg, L

class LaplacianErrorCell(Component): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Laplacian error cell - this is a fixed-point solution
    of a mismatch signal.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: (Unused -- currently cell is a fixed-point model)

        leakRate: (Unused -- currently cell is a fixed-point model)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def lossName(cls):
        return "L"

    @classmethod
    def inputCompartmentName(cls):
        return 'j' ## electrical current

    @classmethod
    def outputCompartmentName(cls):
        return 'e' ## rate-coded output

    @classmethod
    def meanName(cls):
        return 'mu'

    @classmethod
    def derivMeanName(cls):
        return 'dmu'

    @classmethod
    def targetName(cls):
        return 'target'

    @classmethod
    def derivTargetName(cls):
        return 'dtarget'

    @classmethod
    def modulatorName(cls):
        return 'modulator'

    ## Bind Properties to Compartments for ease of use
    @property
    def loss(self):
        return self.compartments.get(self.lossName(), None)

    @loss.setter
    def loss(self, inp):
        self.compartments[self.lossName()] = inp

    @property
    def mean(self):
        return self.compartments.get(self.meanName(), None)

    @mean.setter
    def mean(self, inp):
        self.compartments[self.meanName()] = inp

    @property
    def derivMean(self):
        return self.compartments.get(self.derivMeanName(), None)

    @derivMean.setter
    def derivMean(self, inp):
        self.compartments[self.derivMeanName()] = inp

    @property
    def target(self):
        return self.compartments.get(self.targetName(), None)

    @target.setter
    def target(self, inp):
        self.compartments[self.targetName()] = inp

    @property
    def derivTarget(self):
        return self.compartments.get(self.derivTargetName(), None)

    @derivTarget.setter
    def derivTarget(self, inp):
        self.compartments[self.derivTargetName()] = inp

    @property
    def modulator(self):
        return self.compartments.get(self.modulatorName(), None)

    @modulator.setter
    def modulator(self, inp):
        self.compartments[self.modulatorName()] = inp

    # Define Functions
    def __init__(self, name, n_units, tau_m=0., leakRate=0., key=None,
                 useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1
        self.reset()

    def verify_connections(self):
        self.metadata.check_incoming_connections(self.meanName(), min_connections=1)
        self.metadata.check_incoming_connections(self.targetName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        ## compute Laplacian/MAE error cell output
        self.derivMean, self.derivTarget, self.loss = \
            run_cell(dt, self.target, self.mean)
        if self.modulator is not None:
            self.derivMean = self.derivMean * self.modulator
            self.derivTarget = self.derivTarget * self.modulator
            self.modulator = None ## use and consume modulator

    def reset(self, **kwargs):
        self.derivMean = jnp.zeros((self.batch_size, self.n_units))
        self.derivTarget = jnp.zeros((self.batch_size, self.n_units))
        self.target = jnp.zeros((self.batch_size, self.n_units)) #None
        self.mean = jnp.zeros((self.batch_size, self.n_units)) #None
        self.modulator = None

    def save(self, **kwargs):
        pass
