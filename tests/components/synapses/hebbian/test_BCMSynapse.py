from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
#from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.synapses.hebbian.BCMSynapse import BCMSynapse
from numpy.testing import assert_array_equal

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

        evolve_process = (MethodProcess("evolve_process")
                           >> a.evolve)

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

    pre_value = jnp.ones((1, 1)) * 0.425
    post_value = jnp.ones((1, 1)) * 1.55

    truth = jnp.array([[-1.6798127]])
    reset_process.run()  # ctx.reset()
    a.pre.set(pre_value)
    a.post.set(post_value)
    advance_process.run(t=1., dt=dt)  # ctx.run(t=1., dt=dt)
    evolve_process.run(t=1., dt=dt)  # ctx.adapt(t=1., dt=dt)
    # print(a.dWeights.get())
    # print(truth)
    assert_array_equal(a.dWeights.get(), truth)

test_BCMSynapse1()
