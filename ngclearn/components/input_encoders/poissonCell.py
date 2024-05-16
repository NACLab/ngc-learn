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

@partial(jit, static_argnums=[3])
def sample_poisson(dkey, data, dt, fmax=63.75):
    """
    Samples a Poisson spike train on-the-fly.

    Args:
        dkey: JAX key to drive stochasticity/noise

        data: sensory data (vector/matrix)

        dt: integration time constant

        fmax: maximum frequency (Hz)

    Returns:
        binary spikes
    """
    pspike = data * (dt/1000.) * fmax
    eps = random.uniform(dkey, data.shape, minval=0., maxval=1., dtype=jnp.float32)
    s_t = (eps < pspike).astype(jnp.float32)
    return s_t

class PoissonCell(Component):
    """
    A Poisson cell that produces approximately Poisson-distributed spikes on-the-fly.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        max_freq: maximum frequency (in Hertz) of this Poisson spike train (must be > 0.)

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)
    """

    # Define Functions
    def __init__(self, name, n_units, max_freq=63.75, key=None,
                 useVerboseDict=False, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ## Poisson parameters
        self.max_freq = max_freq ## maximum frequency (in Hertz/Hz)

        ##Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        self.inputs = Compartment(None) # input compartment
        self.outputs = Compartment(jnp.zeros((self.batch_size, self.n_units))) # output compartment
        self.tols = Compartment(jnp.zeros((self.batch_size, self.n_units))) # time of last spike
        self.key = Compartment(random.PRNGKey(time.time_ns()) if key is None else key)

        #self.reset()

    @staticmethod
    def pure_advance(t, dt, max_freq, key, inputs, tols):
        key, *subkeys = random.split(key, 2)
        outputs = sample_poisson(subkeys[0], data=inputs, dt=dt, fmax=max_freq)
        tols = update_times(t, outputs, tols)
        return outputs, tols, key

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
        self.outputs.set(outputs)
        self.tols.set(tols)

    def save(self, **kwargs):
        pass

    # def verify_connections(self):
    #     pass

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

    dkey = random.PRNGKey(1234)
    with Context("Bar") as bar:
        a = PoissonCell("a", n_units=1, max_freq=63.75, key=dkey)
        advance_cmd = AdvanceCommand(components=[a], command_name="Advance")
        reset_cmd = ResetCommand(components=[a], command_name="Reset")

    compiled_advance_cmd, _ = advance_cmd.compile()
    wrapped_advance_cmd = wrapper(jit(compiled_advance_cmd))

    compiled_reset_cmd, _ = reset_cmd.compile()
    wrapped_reset_cmd = wrapper(jit(compiled_reset_cmd))

    T = 50
    dt = 1.

    t = 0. ## global clock
    for i in range(T):
        a.inputs.set(jnp.asarray([[0.5]]))
        wrapped_advance_cmd(t, dt)
        print(f"---[ Step {t} ]---")
        print(f"[a] inputs: {a.inputs.value}, outputs: {a.outputs.value}, time-of-last-spike: {a.tols.value}")
        t += dt
    wrapped_reset_cmd()
    print(f"---[ After reset ]---")
    print(f"[a] inputs: {a.inputs.value}, outputs: {a.outputs.value}, time-of-last-spike: {a.tols.value}")
