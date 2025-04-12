from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import STPDenseSynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
import ngclearn.utils.weight_distribution as dist

def test_STPDenseSynapse1():
    name = "stp_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = STPDenseSynapse(
            name="a", shape=(1,1), resources_init=dist.constant(value=1.),key=subkeys[0]
        )

        #"""
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
        """
    a.weights.set(jnp.ones((1, 1)))
    in_pulse = jnp.ones((1, 1)) * 0.425

    outs_truth = jnp.array([[0.07676563, 0.14312361, 0.16848783]])
    Wdyn_truth = jnp.array([[0.180625,   0.33676142, 0.39644194]])

    outs = []
    Wdyn = []
    ctx.reset()
    for t in range(3):
        a.inputs.set(in_pulse)
        ctx.run(t=t * dt, dt=dt)
        outs.append(a.outputs.value)
        Wdyn.append(a.Wdyn.value)
    outs = jnp.concatenate(outs, axis=1)
    Wdyn = jnp.concatenate(Wdyn, axis=1)
    # print(outs)
    # print(Wdyn)
    np.testing.assert_allclose(outs, outs_truth, atol=1e-8)
    np.testing.assert_allclose(Wdyn, Wdyn_truth, atol=1e-8)

#test_STPDenseSynapse1()
