from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

#@partial(jit, static_argnums=[3])
def run_cell(dt, targ, mu):
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
    of a mismatch/error signal.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: (Unused -- currently cell is a fixed-point model)

        leakRate: (Unused -- currently cell is a fixed-point model)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)
    """

    # Define Functions
    def __init__(self, name, n_units, tau_m=0., leakRate=0., key=None,
                 useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number setup
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##Layer Size setup
        self.n_units = n_units
        self.batch_size = 1

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.mean = Compartment(restVals)
        self.target = Compartment(restVals)
        self.derivMean = Compartment(restVals)
        self.derivTarget = Compartment(restVals)
        self.loss = Compartment(jnp.zeros((1,1)))
        self.modulator = Compartment(jnp.ones((1,1)))
        self.reset()

    @staticmethod
    def pure_advance(t, dt, target, mean, derivTarget, derivMean, modulator):
        ## compute Laplacian/MAE error cell output
        derivMean, derivTarget, loss = run_cell(dt, target, mean)
        #if modulator is not None:
        derivMean = derivMean * modulator
        derivTarget = derivTarget * modulator
        #modulator = None ## use and consume modulator

    @resolver(pure_advance, output_compartments=['target', 'mean', 'derivTarget', 'derivMean', 'modulator'])
    def advance(self, vals):
        target, mean, derivTarget, derivMean, modulator = vals
        self.target.set(target)
        self.mean.set(mean)
        self.derivTarget.set(derivTarget)
        self.derivMean.set(derivMean)
        self.modulator.set(modulator)

    @staticmethod
    def pure_reset(batch_size, n_units):
        restVals = [jnp.zeros((batch_size, n_units)) for _ in range(4)] + [jnp.ones((batch_size, n_units))]
        return restVals

    @resolver(pure_reset, output_compartments=['target', 'mean', 'derivTarget', 'derivMean', 'modulator'])
    def reset(self, vals):
        target, mean, derivTarget, derivMean, modulator = vals
        self.target.set(target)
        self.mean.set(mean)
        self.derivTarget.set(derivTarget)
        self.derivMean.set(derivMean)
        self.modulator.set(modulator)

    # def save(self, **kwargs):
    #     pass

    # def verify_connections(self):
    #     self.metadata.check_incoming_connections(self.meanName(), min_connections=1)
    #     self.metadata.check_incoming_connections(self.targetName(), min_connections=1)
