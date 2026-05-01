from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
#import ngclearn.utils.weight_distribution as dist
from ngclearn.components.synapses.modulated.MSTDPETSynapse import MSTDPETSynapse
from numpy.testing import assert_array_equal

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

        evolve_process = (MethodProcess("evolve_process")
                          >> a.evolve)

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

    a.weights.set(jnp.ones((1, 1)) * 0.75)

    in_spike = jnp.ones((1, 1))
    in_trace = jnp.ones((1, 1,)) * 1.25
    out_spike = jnp.ones((1, 1))
    out_trace = jnp.ones((1, 1,)) * 0.65
    r_neg = -jnp.ones((1, 1))
    r_pos = jnp.ones((1, 1))

    #print(a.weights.value)
    reset_process.run()  # ctx.reset()
    a.preSpike.set(in_spike * 0)
    a.preTrace.set(in_trace)
    a.postSpike.set(out_spike)
    a.postTrace.set(out_trace)
    a.modulator.set(r_pos)
    advance_process.run(t=1., dt=dt)  # ctx.run(t=1. * dt, dt=dt)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1. * dt, dt=dt)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1. * dt, dt=dt)
    #print(a.weights.get())
    assert_array_equal(a.weights.get(), jnp.array([[0.875]]))

    reset_process.run()  # ctx.reset()
    a.preSpike.set(in_spike * 0)
    a.preTrace.set(in_trace)
    a.postSpike.set(out_spike)
    a.postTrace.set(out_trace)
    a.modulator.set(r_neg)
    advance_process.run(t=1., dt=dt)  # ctx.run(t=1. * dt, dt=dt)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1. * dt, dt=dt)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1. * dt, dt=dt)
    #print(a.weights.get())
    assert_array_equal(a.weights.get(), jnp.array([[0.75]]))

#test_MSTDPETSynapse1()
