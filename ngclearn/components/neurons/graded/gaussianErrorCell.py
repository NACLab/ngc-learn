from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver

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
    return run_gaussian_cell(dt, targ, mu)

@jit
def run_gaussian_cell(dt, targ, mu):
    """
    Moves Gaussian cell dynamics one step forward. Specifically, this
    routine emulates the error unit behavior of the local cost functional:

    | L(targ, mu) = -(1/2) * ||targ - mu||^2_2
    | or log likelihood of the multivariate Gaussian with identity covariance

    Args:
        dt: integration time constant

        targ: target pattern value

        mu: prediction value

    Returns:
        derivative w.r.t. mean "dmu", derivative w.r.t. target dtarg, loss
    """
    dmu = (targ - mu) # e (error unit)
    dtarg = -dmu # reverse of e
    L = -jnp.sum(jnp.square(dmu)) * 0.5
    return dmu, dtarg, L

class GaussianErrorCell(Component): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Gaussian error cell - this is a fixed-point solution
    of a mismatch signal.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: (Unused -- currently cell is a fixed-point model)

        leakRate: (Unused -- currently cell is a fixed-point model)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, tau_m=0., leakRate=0., key=None):
        super().__init__(name)

        ##Random Number Set up
        self.key = Compartment(random.PRNGKey(time.time_ns()) if key is None else key)
        self.j = Compartment(None) # ## electrical current/ input compartment/to be wired/set. # NOTE: VN: This is never used
        self.L = Compartment(None) # loss compartment
        self.e = Compartment(None) # rate-coded output/ output compartment/to be wired/set. # NOTE: VN: This is never used
        self.mu = Compartment(jnp.zeros((self.batch_size, self.n_units))) # mean/mean name
        self.dmu = Compartment(jnp.zeros((self.batch_size, self.n_units))) # derivative mean
        self.target = Compartment(jnp.zeros((self.batch_size, self.n_units))) # target
        self.dtarget = Compartment(jnp.zeros((self.batch_size, self.n_units))) # derivative target
        self.modulator = Compartment(jnp.asarray(0.0)) # to be set/consumed

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

    # def verify_connections(self):
    #     self.metadata.check_incoming_connections(self.meanName(), min_connections=1)
    #     self.metadata.check_incoming_connections(self.targetName(), min_connections=1)

    @staticmethod
    def pure_advance(t, dt, mu, dmu, target, dtarget, modulator):
        ## compute Gaussian error cell output
        dmu, dtarget, L = run_cell(dt, target, mu)
        modulator_mask = jnp.bool(modulator).astype(jnp.float32)
        dmu = dmu * (1 - modulator_mask) + dmu * modulator * modulator_mask
        dtarget = dtarget * (1 - modulator_mask) + dtarget * modulator * modulator_mask
        modulator = jnp.asarray(0.0) ## use and consume modulator
        return dmu, dtarget, L, modulator


    @resolver(pure_advance, output_compartments=['dmu', 'dtarget', 'L', 'modulator'])
    def advance(self, dmu, dtarget, L, modulator):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.L.set(L)
        self.modulator.set(modulator)



    def reset(self, **kwargs):
        self.derivMean = jnp.zeros((self.batch_size, self.n_units))
        self.derivTarget = jnp.zeros((self.batch_size, self.n_units))
        self.target = jnp.zeros((self.batch_size, self.n_units)) #None
        self.mean = jnp.zeros((self.batch_size, self.n_units)) #None
        self.modulator = None

    def save(self, **kwargs):
        pass
