from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver

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
    def __init__(self, name, n_units, tau_m=0., leakRate=0., key=None, **kwargs):
        super().__init__(name, **kwargs)

        ##Layer Size setup
        self.n_units = n_units
        self.batch_size = 1

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(None) # ## electrical current/ input compartment/to be wired/set. # NOTE: VN: This is never used
        self.L = Compartment(None) # loss compartment
        self.e = Compartment(None) # rate-coded output/ output compartment/to be wired/set. # NOTE: VN: This is never used
        self.mu = Compartment(restVals) # mean/mean name. input wire
        self.dmu = Compartment(restVals) # derivative mean
        self.target = Compartment(restVals) # target. input wire
        self.dtarget = Compartment(restVals) # derivative target
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed

    @staticmethod
    def _advance_state(t, dt, mu, dmu, target, dtarget, modulator):
        ## compute Laplacian error cell output
        dmu, dtarget, L = run_cell(dt, target, mu)
        dmu = dmu * modulator
        dtarget = dtarget * modulator
        return dmu, dtarget, L

    @resolver(_advance_state)
    def advance_state(self, dmu, dtarget, L):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.L.set(L)

    @staticmethod
    def _reset(batch_size, n_units):
        dmu = jnp.zeros((batch_size, n_units))
        dtarget = jnp.zeros((batch_size, n_units))
        target = jnp.zeros((batch_size, n_units)) #None
        mu = jnp.zeros((batch_size, n_units)) #None
        return dmu, dtarget, target, mu

    @resolver(_reset)
    def reset(self, dmu, dtarget, target, mu):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.target.set(target)
        self.mu.set(mu)
        self.modulator.set(mu + 1.)
