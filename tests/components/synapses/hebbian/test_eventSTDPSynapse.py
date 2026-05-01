from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
#from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.synapses.hebbian.eventSTDPSynapse import EventSTDPSynapse
from numpy.testing import assert_array_equal

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

        evolve_process = (MethodProcess("evolve_process")
                          >> a.evolve)

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

    a.weights.set(jnp.ones((1, 1)) * 0.1)

    t = 12. ## fake out current time
    ## Case 1: outside pre-syn time window
    input_tols = jnp.ones((1, 1,)) * 9.
    out_spike = jnp.ones((1, 1))

    ## check pre-synaptic STDP only
    truth = jnp.array([[-0.101]])
    reset_process.run()  # ctx.reset()
    a.pre_tols.set(input_tols)
    a.postSpike.set(out_spike)
    advance_process.run(t=t, dt=dt)  # ctx.run(t=t, dt=dt)
    evolve_process.run(t=t, dt=dt)  # ctx.adapt(t=t, dt=dt)
    # print(a.dWeights.get())
    # print(truth)
    assert_array_equal(a.dWeights.get(), truth)

    ## Case 2: within pre-syn time window
    input_tols = jnp.ones((1, 1,)) * 11.
    out_spike = jnp.ones((1, 1))

    ## check pre-synaptic STDP only
    truth = jnp.array([[0.899]])
    reset_process.run()  # ctx.reset()
    a.pre_tols.set(input_tols)
    a.postSpike.set(out_spike)
    advance_process.run(t=t, dt=dt)  # ctx.run(t=t, dt=dt)
    evolve_process.run(t=t, dt=dt)  # ctx.adapt(t=t, dt=dt)
    # print(a.dWeights.get())
    # print(truth)
    assert_array_equal(a.dWeights.get(), truth)

#test_eventSTDPSynapse1()

