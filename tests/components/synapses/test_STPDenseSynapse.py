from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngclearn.components.synapses.STPDenseSynapse import STPDenseSynapse

def test_STPDenseSynapse1():
    name = "stp_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = STPDenseSynapse(
            name="a", shape=(1,1), resources_init=DistributionGenerator.constant(value=1.),key=subkeys[0]
        )

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

    a.weights.set(jnp.ones((1, 1)))
    in_pulse = jnp.ones((1, 1)) * 0.425

    outs_truth = jnp.array([[0.07676563, 0.14312361, 0.16848783]])
    Wdyn_truth = jnp.array([[0.180625,   0.33676142, 0.39644194]])

    outs = []
    Wdyn = []
    reset_process.run()  # ctx.reset()
    for t in range(3):
        a.inputs.set(in_pulse)
        advance_process.run(t=t * 1., dt=dt)  # ctx.run(t=ts * 1., dt=dt)
        outs.append(a.outputs.get())
        Wdyn.append(a.Wdyn.get())
    outs = jnp.concatenate(outs, axis=1)
    Wdyn = jnp.concatenate(Wdyn, axis=1)
    # print(outs)
    # print(outs_truth)
    # print("...")
    # print(Wdyn)
    # print(Wdyn_truth)
    np.testing.assert_allclose(outs, outs_truth, atol=1e-8)
    np.testing.assert_allclose(Wdyn, Wdyn_truth, atol=1e-8)

#test_STPDenseSynapse1()
