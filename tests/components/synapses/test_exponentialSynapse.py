from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import ExponentialSynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process
from ngcsimlib.context import Context
import ngclearn.utils.weight_distribution as dist

def test_exponentialSynapse1():
    name = "expsyn_ctx"
    ## create seeding keys
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    ## excitatory properties
    tau_syn = 2.
    E_rest = 0.
    ## inhibitory properties
    #tau_syn = 5.
    #E_rest = -80.
    # ---- build a single exp-synapse system ----
    with Context(name) as ctx:
        a = ExponentialSynapse(
            name="a", shape=(1,1), tau_syn=tau_syn, g_syn_bar=2.4, syn_rest=E_rest, weight_init=dist.constant(value=1.),
            key=subkeys[0]
        )

        advance_process = (Process("advance_proc")
                           >> a.advance_state)
        # ctx.wrap_and_add_command(advance_process.pure, name="run")
        ctx.wrap_and_add_command(jit(advance_process.pure), name="run")

        reset_process = (Process("reset_proc")
                         >> a.reset)
        ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")

    sp_train = jnp.array([1., 0., 1.], dtype=jnp.float32)
    post_syn_neuron_volt = jnp.ones((1, 1)) * -65. ## post-syn neuron is at rest

    outs_truth = jnp.array([[156.,  78., 195.]])

    outs = []
    ctx.reset()
    for t in range(3):
        in_pulse = jnp.expand_dims(sp_train[t], axis=0)
        a.inputs.set(in_pulse)
        a.v.set(post_syn_neuron_volt)
        ctx.run(t=t * dt, dt=dt)
        print("g: ",a.g_syn.value)
        print("i: ", a.i_syn.value)
        outs.append(a.outputs.value)
    outs = jnp.concatenate(outs, axis=1)
    #print(outs)

    np.testing.assert_allclose(outs, outs_truth, atol=1e-8)

test_exponentialSynapse1()
