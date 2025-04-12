from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import MSTDPETSynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
import ngclearn.utils.weight_distribution as dist

def test_MSTDPETSynapse1():
    name = "mstdpet_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = MSTDPETSynapse(
            name="a", shape=(1,1), A_plus=1., A_minus=1., eta=0.1, key=subkeys[0]
        )

        #"""
        advance_process = (Process("advance_proc")
                           >> a.advance_state)
        # ctx.wrap_and_add_command(advance_process.pure, name="run")
        ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        evolve_process = (Process("evolve_proc")
                           >> a.evolve)
        #ctx.wrap_and_add_command(evolve_process.pure, name="run")
        ctx.wrap_and_add_command(jit(evolve_process.pure), name="adapt")
        
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

    a.weights.set(jnp.ones((1, 1)) * 0.75)

    in_spike = jnp.ones((1, 1))
    in_trace = jnp.ones((1, 1,)) * 1.25
    out_spike = jnp.ones((1, 1))
    out_trace = jnp.ones((1, 1,)) * 0.65
    r_neg = -jnp.ones((1, 1))
    r_pos = jnp.ones((1, 1))

    #print(a.weights.value)
    ctx.reset()
    a.preSpike.set(in_spike * 0)
    a.preTrace.set(in_trace)
    a.postSpike.set(out_spike)
    a.postTrace.set(out_trace)
    a.modulator.set(r_pos)
    ctx.run(t=1. * dt, dt=dt)
    ctx.adapt(t=1. * dt, dt=dt)
    ctx.adapt(t=1. * dt, dt=dt)
    #print(a.weights.value)
    assert_array_equal(a.weights.value, jnp.array([[0.875]]))

    ctx.reset()
    a.preSpike.set(in_spike * 0)
    a.preTrace.set(in_trace)
    a.postSpike.set(out_spike)
    a.postTrace.set(out_trace)
    a.modulator.set(r_neg)
    ctx.run(t=1. * dt, dt=dt)
    ctx.adapt(t=1. * dt, dt=dt)
    ctx.adapt(t=1. * dt, dt=dt)
    #print(a.weights.value)
    assert_array_equal(a.weights.value, jnp.array([[0.75]]))

#test_MSTDPETSynapse1()
