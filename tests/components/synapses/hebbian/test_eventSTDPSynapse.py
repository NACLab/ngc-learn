from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import EventSTDPSynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context

def test_eventSTDPSynapse1():
    name = "event_stdp_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    trace_increment = 0.1
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = EventSTDPSynapse(
            name="a", shape=(1,1), eta=0., presyn_win_len=2., key=subkeys[0]
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
    a.weights.set(jnp.ones((1, 1)) * 0.1)

    t = 12. ## fake out current time
    ## Case 1: outside of pre-syn time window
    input_tols = jnp.ones((1, 1,)) * 9.
    out_spike = jnp.ones((1, 1))

    ## check pre-synaptic STDP only
    truth = jnp.array([[-0.101]])
    ctx.reset()
    a.pre_tols.set(input_tols)
    a.postSpike.set(out_spike)
    ctx.run(t=t, dt=dt)
    ctx.adapt(t=t, dt=dt)
    #print(a.dWeights.value)
    assert_array_equal(a.dWeights.value, truth)

    ## Case 2: within pre-syn time window
    input_tols = jnp.ones((1, 1,)) * 11.
    out_spike = jnp.ones((1, 1))

    ## check pre-synaptic STDP only
    truth = jnp.array([[0.899]])
    ctx.reset()
    a.pre_tols.set(input_tols)
    a.postSpike.set(out_spike)
    ctx.run(t=t, dt=dt)
    ctx.adapt(t=t, dt=dt)
    #print(a.dWeights.value)
    assert_array_equal(a.dWeights.value, truth)

#test_eventSTDPSynapse1()

