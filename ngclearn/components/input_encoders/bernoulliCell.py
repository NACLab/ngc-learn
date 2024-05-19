# %%

from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver

from jax import numpy as jnp, random, jit
from functools import partial
import time

@jit
def update_times(t, s, tols):
    """
    Updates time-of-last-spike (tols) variable.

    Args:
        t: current time (a scalar/int value)

        s: binary spike vector

        tols: current time-of-last-spike variable

    Returns:
        updated tols variable
    """
    _tols = (1. - s) * tols + (s * t)
    return _tols

@jit
def sample_bernoulli(dkey, data):
    """
    Samples a Bernoulli spike train on-the-fly

    Args:
        dkey: JAX key to drive stochasticity/noise

        data: sensory data (vector/matrix)

    Returns:
        binary spikes
    """
    s_t = random.bernoulli(dkey, p=data).astype(jnp.float32)
    return s_t

class BernoulliCell(Component):
    """
    A Bernoulli cell that produces Bernoulli-distributed spikes on-the-fly.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell
    """

    # Define Functions
    def __init__(self, name, n_units, key=None):
        super().__init__(name)

        ##Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.inputs = Compartment(None) # input compartment
        self.outputs = Compartment(jnp.zeros((self.batch_size, self.n_units))) # output compartment
        self.tols = Compartment(jnp.zeros((self.batch_size, self.n_units))) # time of last spike
        self.key = Compartment(random.PRNGKey(time.time_ns()) if key is None else key)

    @staticmethod
    def pure_advance(t, dt, key, inputs, tols):
        key, *subkeys = random.split(key, 2)
        outputs = sample_bernoulli(subkeys[0], data=inputs)
        timeOfLastSpike = update_times(t, outputs, tols)
        # print(f"[BernoulliCell.pure_advance] inputs: {inputs.shape}, outputs: {outputs.shape}")
        return outputs, timeOfLastSpike, key

    @resolver(pure_advance, output_compartments=['outputs', 'tols', 'key'])
    def advance(self, vals):
        outputs, tols, key = vals
        self.outputs.set(outputs)
        self.tols.set(tols)
        self.key.set(key)

    @staticmethod
    def pure_reset(batch_size, n_units):
        return None, jnp.zeros((batch_size, n_units)), jnp.zeros((batch_size, n_units))

    @resolver(pure_reset, output_compartments=['inputs', 'outputs', 'tols'])
    def reset(self, inputs, outputs, tols):
        self.inputs.set(inputs)
        self.outputs.set(outputs) #None
        self.tols.set(tols)

    def save(self, directory, **kwargs):
        pass

    def load(self, directory, **kwargs):
        pass


# Testing
if __name__ == '__main__':
    from ngcsimlib.compartment import All_compartments
    from ngcsimlib.context import Context
    from ngcsimlib.commands import Command
    from ngclearn.components.neurons.graded.rateCell import RateCell

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
                component.reset(t=t, dt=dt)

    with Context("Bar") as bar:
        b = BernoulliCell("b", 2)
        a = RateCell("a2", 2, 0.01)
        a.j << b.outputs
        advance_cmd = AdvanceCommand(components=[b, a], command_name="Advance")
        reset_cmd = ResetCommand(components=[b, a], command_name="Reset")

    compiled_advance_cmd, _ = advance_cmd.compile()
    wrapped_advance_cmd = wrapper(jit(compiled_advance_cmd))

    compiled_reset_cmd, _ = reset_cmd.compile()
    wrapped_reset_cmd = wrapper(jit(compiled_reset_cmd))

    dt = 0.01
    for t in range(5):
        b.inputs.set(jnp.asarray([[0.5, 0.5]]))
        wrapped_advance_cmd(t, dt)
        print(f"---[ Step {t} ]---")
        print(f"[b] inputs: {b.inputs.value}, outputs: {b.outputs.value}, time of last spike: {b.tols.value}")
        print(f"[a] j: {a.j.value}, j_td: {a.j_td.value}, z: {a.z.value}, zF: {a.zF.value}")
    wrapped_reset_cmd()
    print(f"---[ After reset ]---")
    print(f"[b] inputs: {b.inputs.value}, outputs: {b.outputs.value}, time of last spike: {b.tols.value}")
    print(f"[a] j: {a.j.value}, j_td: {a.j_td.value}, z: {a.z.value}, zF: {a.zF.value}")

