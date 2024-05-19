# %%

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

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

        ##Random Number Set up
        self.key = Compartment(random.PRNGKey(time.time_ns()) if key is None else key)
        self.j = Compartment(None) # ## electrical current/ input compartment/to be wired/set. # NOTE: VN: This is never used
        self.L = Compartment(None) # loss compartment
        self.e = Compartment(None) # rate-coded output/ output compartment/to be wired/set. # NOTE: VN: This is never used
        self.mu = Compartment(jnp.zeros((self.batch_size, self.n_units))) # mean/mean name. input wire
        self.dmu = Compartment(jnp.zeros((self.batch_size, self.n_units))) # derivative mean
        self.target = Compartment(jnp.zeros((self.batch_size, self.n_units))) # target. input wire
        self.dtarget = Compartment(jnp.zeros((self.batch_size, self.n_units))) # derivative target
        self.modulator = Compartment(jnp.asarray(0.0)) # to be set/consumed

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

    @staticmethod
    def pure_reset(batch_size, n_units):
        dmu = jnp.zeros((batch_size, n_units))
        dtarget = jnp.zeros((batch_size, n_units))
        target = jnp.zeros((batch_size, n_units)) #None
        mu = jnp.zeros((batch_size, n_units)) #None
        modulator = jnp.asarray(0.0)
        return dmu, dtarget, target, mu, modulator

    @resolver(pure_reset, output_compartments=['dmu', 'dtarget', 'target', 'mu', 'modulator'])
    def reset(self, dmu, dtarget, target, mu, modulator):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.target.set(target)
        self.mu.set(mu)
        self.modulator.set(modulator)

    def save(self, directory, **kwargs):
        pass

    def load(self, directory, **kwargs):
        pass

# Testing
if __name__ == '__main__':
    from ngcsimlib.compartment import All_compartments
    from ngcsimlib.context import Context
    from ngcsimlib.commands import Command

    def wrapper(compiled_fn):
        def _wrapped(*args):
            # vals = jax.jit(compiled_fn)(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            vals = compiled_fn(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            for key, value in vals.items():
                All_compartments[str(key)].set(value)
            return vals
        return _wrapped

    class AdvanceCommand(Command):
        compile_key = "advance"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.gather()
                component.advance(t=t, dt=dt)

    class ResetCommand(Command):
        compile_key = "reset"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.gather()
                component.reset()

    with Context("Bar") as bar:
        e = GaussianErrorCell("e", 2)
        advance_cmd = AdvanceCommand(components=[e], command_name="Advance")
        reset_cmd = ResetCommand(components=[e], command_name="Reset")

    compiled_advance_cmd, _ = advance_cmd.compile()
    # wrapped_advance_cmd = wrapper(jit(compiled_advance_cmd))
    wrapped_advance_cmd = wrapper(compiled_advance_cmd)

    compiled_reset_cmd, _ = reset_cmd.compile()
    wrapped_reset_cmd = wrapper(compiled_reset_cmd)

    dt = 20.0
    for t in range(4):
        e.mu.set(jnp.asarray([[0.1, 0.9]]))
        e.target.set(jnp.asarray([[1.0, -1.0]]))
        wrapped_advance_cmd(t, dt)
        print(f"Step {t} - [e] mu: {e.mu.value}, target: {e.target.value}, dmu: {e.dmu.value}, dtarget: {e.dtarget.value}, L: {e.L.value}, modulator: {e.modulator.value}")
    wrapped_reset_cmd()
    print(f"Step {t} - [e] mu: {e.mu.value}, target: {e.target.value}, dmu: {e.dmu.value}, dtarget: {e.dtarget.value}, L: {e.L.value}, modulator: {e.modulator.value}")

