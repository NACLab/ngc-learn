from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngclearn.components.synapses.exponentialSynapse import ExponentialSynapse

def test_exponentialSynapse1():
    name = "expsyn_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    ## excitatory properties
    tau_syn = 2.
    E_rest = 0.
    # ---- build a single exp-synapse system ----
    with Context(name) as ctx:
        a = ExponentialSynapse(
            name="a", shape=(1,1), tau_decay=tau_syn, g_syn_bar=2.4, syn_rest=E_rest, 
            weight_init=DistributionGenerator.constant(value=1.), key=subkeys[0]
        )

        advance_process = (MethodProcess("advance_proc")
                           >> a.advance_state)

        reset_process = (MethodProcess("reset_proc")
                         >> a.reset)

    sp_train = jnp.array([1., 0., 1.], dtype=jnp.float32)
    post_syn_neuron_volt = jnp.ones((1, 1)) * -65. ## post-syn neuron is at rest

    outs_truth = jnp.array([[156.,  78., 195.]])

    outs = []
    reset_process.run()  # ctx.reset()
    for t in range(3):
        in_pulse = jnp.expand_dims(sp_train[t], axis=0)
        a.inputs.set(in_pulse)
        a.v.set(post_syn_neuron_volt)
        advance_process.run(t=t * 1., dt=dt)  # ctx.run(t=ts * 1., dt=dt)
        # print("in: ", a.inputs.get())
        # print("g: ",a.g_syn.get())
        # print("i: ", a.i_syn.get())
        outs.append(a.outputs.get())
    outs = jnp.concatenate(outs, axis=1)
    #print(outs)

    np.testing.assert_allclose(outs, outs_truth, atol=1e-8)

#test_exponentialSynapse1()
