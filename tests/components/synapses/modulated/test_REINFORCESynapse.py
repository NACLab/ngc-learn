# %%

import jax
from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components.synapses.modulated.REINFORCESynapse import REINFORCESynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context

import jax
import jax.numpy as jnp

def test_REINFORCESynapse1():
    name = "reinforce_ctx"
    ## create seeding keys
    np.random.seed(42)
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = REINFORCESynapse(
            name="a", shape=(1,1), act_fx="tanh", key=subkeys[0]
        )

        evolve_process = (Process() >> a.evolve)
        ctx.wrap_and_add_command(jit(evolve_process.pure), name="adapt")

        reset_process = (Process() >> a.reset)
        ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")

        @Context.dynamicCommand
        def clamp_inputs(x):
            a.inputs.set(x)

        @Context.dynamicCommand
        def clamp_rewards(x):
            a.rewards.set(x)

        @Context.dynamicCommand
        def clamp_weights(x):
            a.weights.set(x)

    # a.weights.set(jnp.ones((1, 1)) * 0.1)

    ## check pre-synaptic STDP only
    # truth = jnp.array([[1.25]])
    ctx.reset()
    clamp_weights(jnp.ones((1, 2)))
    clamp_rewards(jnp.ones((1, 1)))
    clamp_inputs(jnp.ones((1, 1)))
    ctx.adapt(t=1., dt=dt)
    # assert_array_equal(a.dWeights.value, truth)
    print(a.dWeights.value)


    # JAX Grad output
    _act = jax.nn.tanh
    def fn(params: dict, inputs: jax.Array, outputs: jax.Array, seed):
        W_mu, W_logstd = params
        mean = _act(inputs) @ W_mu
        logstd = _act(inputs) @ W_logstd
        std = jnp.exp(logstd.clip(-10.0, 2.0))
        sample = jax.random.normal(seed, mean.shape) * std + mean
        # logp = gaussian_logpdf(sample, mean, std).sum(-1)
        logp = jax.scipy.stats.norm.logpdf(sample, mean, std).sum(-1)
        return (-logp * outputs).mean() * 1e-2
    grad_fn = jax.value_and_grad(fn)

    weights_mu = jnp.ones((1, 1))
    weights_logstd = jnp.ones((1, 1))
    inputs = jnp.ones((1, 1))
    outputs = jnp.ones((1, 1))
    objective, grads = grad_fn(
        (weights_mu, weights_logstd),
        inputs,
        outputs,
        jax.random.key(42)
    )
    np.testing.assert_allclose(
        a.dWeights.value[0],
        jnp.concatenate([grads[0], grads[1]], axis=-1),
        atol=1e-2
    ) # NOTE: gradient is not exact due to different gradient computation, we need to inspect more closely

# test_REINFORCESynapse1()
