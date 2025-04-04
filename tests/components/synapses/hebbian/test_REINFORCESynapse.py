# %%

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components.synapses.hebbian.REINFORCESynapse import REINFORCESynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context

def test_REINFORCESynapse1():
    name = "reinforce_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = REINFORCESynapse(
            name="a", shape=(1,1), act_fx="tanh", key=subkeys[0]
        )

        evolve_process = (Process() >> a.evolve)
        ctx.wrap_and_add_command(jit(evolve_process.pure), name="adapt")

        reset_process = (Process() >> a.reset)
        ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")

        @Context.dynamicCommand
        def clamp_inputs(x):
            a.inputs.set(x)

        @Context.dynamicCommand
        def clamp_rewards(x):
            a.rewards.set(x)

    # a.weights.set(jnp.ones((1, 1)) * 0.1)

    ## check pre-synaptic STDP only
    # truth = jnp.array([[1.25]])
    ctx.reset()
    clamp_rewards(jnp.ones((1, 1)))
    clamp_inputs(jnp.ones((1, 1)))
    ctx.adapt(t=1., dt=dt)
    # assert_array_equal(a.dWeights.value, truth)
    print(a.dWeights.value)

# test_REINFORCESynapse1()

