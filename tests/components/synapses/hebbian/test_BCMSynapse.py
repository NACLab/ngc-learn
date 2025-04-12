from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import BCMSynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context

def test_BCMSynapse1():
    name = "bcm_stdp_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = BCMSynapse(
            name="a", shape=(1,1), tau_w=40., tau_theta=20., key=subkeys[0]
        )

        #"""
        evolve_process = (Process("evolve_proc")
                           >> a.evolve)
        #ctx.wrap_and_add_command(evolve_process.pure, name="run")
        ctx.wrap_and_add_command(jit(evolve_process.pure), name="adapt")

        advance_process = (Process("advance_proc")
                           >> a.advance_state)
        # ctx.wrap_and_add_command(advance_process.pure, name="run")
        ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        reset_process = (Process("reset_proc")
                         >> a.reset)
        ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
        #"""

        """
        reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
        ctx.add_command(wrap_command(jit(ctx.reset)), name="reset")
        advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
        ctx.add_command(wrap_command(jit(ctx.advance_state)), name="run")
        evolve_cmd, evolve_args = ctx.compile_by_key(a, compile_key="evolve")
        ctx.add_command(wrap_command(jit(ctx.evolve)), name="adapt")
        """

    pre_value = jnp.ones((1, 1)) * 0.425
    post_value = jnp.ones((1, 1)) * 1.55

    truth = jnp.array([[-1.6798127]])
    ctx.reset()
    a.pre.set(pre_value)
    a.post.set(post_value)
    ctx.run(t=1., dt=dt)
    ctx.adapt(t=1., dt=dt)
    #print(a.dWeights.value)
    assert_array_equal(a.dWeights.value, truth)


#test_BCMSynapse1()
