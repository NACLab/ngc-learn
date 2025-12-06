
from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
#from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.synapses.hebbian.expSTDPSynapse import ExpSTDPSynapse
from numpy.testing import assert_array_equal

def test_expSTDPSynapse1():
    name = "exp_stdp_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = ExpSTDPSynapse(
            name="a", shape=(1,1), A_plus=1., A_minus=1., exp_beta=1.25, eta=0., key=subkeys[0]
        )

        evolve_process = (MethodProcess("evolve_process")
                          >> a.evolve)

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

    a.weights.set(jnp.ones((1, 1)) * 0.1)

    in_spike = jnp.ones((1, 1))
    in_trace = jnp.ones((1, 1,)) * 1.25
    out_spike = jnp.ones((1, 1))
    out_trace = jnp.ones((1, 1,)) * 0.65

    ## check pre-synaptic STDP only
    truth = jnp.array([[1.1031212]])
    reset_process.run()  # ctx.reset()
    a.preSpike.set(in_spike * 0)
    a.preTrace.set(in_trace)
    a.postSpike.set(out_spike)
    a.postTrace.set(out_trace)
    advance_process.run(t=1., dt=dt)  # ctx.run(t=1., dt=dt)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1., dt=dt)
    # print("W: ",a.weights.get())
    # print(a.dWeights.get())
    # print(truth)
    assert_array_equal(a.dWeights.get(), truth)

    truth = jnp.array([[-0.57362294]])
    reset_process.run()  # ctx.reset()
    a.preSpike.set(in_spike)
    a.preTrace.set(in_trace)
    a.postSpike.set(out_spike * 0)
    a.postTrace.set(out_trace)
    advance_process.run(t=1., dt=dt)  # ctx.run(t=1., dt=dt)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1., dt=dt)
    # print("W: ", a.weights.get())
    # print(a.dWeights.get())
    # print(truth)
    assert_array_equal(a.dWeights.get(), truth)

#test_expSTDPSynapse1()

