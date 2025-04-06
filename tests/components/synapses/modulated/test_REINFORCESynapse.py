# %%

import jax
from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components.synapses.modulated.REINFORCESynapse import REINFORCESynapse, gaussian_logpdf
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
    decay = 0.99
    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = REINFORCESynapse(
            name="a", shape=(1,1), decay=decay, act_fx="tanh", key=subkeys[0]
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

    # Function definition
    _act = jax.nn.tanh
    def fn(params: dict, inputs: jax.Array, outputs: jax.Array):
        W_mu, W_logstd = params
        activation = _act(inputs)
        mean = activation @ W_mu
        logstd = activation @ W_logstd
        std = jnp.exp(logstd.clip(-10.0, 2.0))
        # sample = jax.random.normal(seed, mean.shape) * std + mean
        sample = jnp.asarray(np.random.normal(0, 1, mean.shape)) * std + mean
        logp = gaussian_logpdf(sample, mean, std).sum(-1)
        # logp = jax.scipy.stats.norm.logpdf(sample, mean, std).sum(-1)
        return (-logp * outputs).mean() * 1e-2
    grad_fn = jax.value_and_grad(fn)

    # Some setups
    expected_weights_mu = jnp.asarray([[0.13]])
    expected_weights_logstd = jnp.asarray([[0.04]])
    expected_weights = jnp.concatenate([expected_weights_mu, expected_weights_logstd], axis=-1)
    initial_ngclearn_weights = jnp.concatenate([expected_weights_mu, expected_weights_logstd], axis=-1)[None]
    expected_gradient_list = []
    ctx.reset()

    # Loop through 3 steps
    step = 1
    # ---------------- Step {step} --------------------
    print(f"------------ [Step {step}] ------------")
    inputs = -1**step * jnp.ones((1, 1)) / 10  # * 0.5 * step / 10.0
    outputs = -1**step * jnp.ones((1, 1)) / 10 # * 3 * step / 10.0# reward
    # --------- ngclearn ---------
    clamp_weights(initial_ngclearn_weights)
    clamp_rewards(outputs)
    clamp_inputs(inputs)
    np.random.seed(42)
    ctx.adapt(t=1., dt=dt)
    print(f"[ngclearn] objective: {a.objective.value}")
    print(f"[ngclearn] weights: {a.weights.value}")
    print(f"[ngclearn] dWeights: {a.dWeights.value}")
    print(f"[ngclearn] step_count: {a.step_count.value}")
    print(f"[ngclearn] accumulated_gradients: {a.accumulated_gradients.value}")
    # -------- Expectation ---------
    print("--------------")
    np.random.seed(42)
    expected_objective, expected_grads = grad_fn(
        (expected_weights_mu, expected_weights_logstd),
        inputs,
        outputs,
    )
    # NOTE: Viet: negate the gradient because gradient in ngc-learn
    #   is gradient ascent, while gradient in JAX is gradient descent
    expected_grads = -jnp.concatenate([expected_grads[0], expected_grads[1]], axis=-1)
    expected_gradient_list.append(expected_grads)
    print(f"[Expectation] expected_weights: {expected_weights}")
    print(f"[Expectation] dWeights: {expected_grads}")
    print(f"[Expectation] objective: {expected_objective}")
    np.testing.assert_allclose(
        a.dWeights.value[0],
        expected_grads,
        atol=1e-8
    )
    np.testing.assert_allclose(
        a.objective.value,
        expected_objective,
        atol=1e-8
    )
    print()


    step = 2
    # ---------------- Step {step} --------------------
    print(f"------------ [Step {step}] ------------")
    inputs = -1**step * jnp.ones((1, 1)) / 10  # * 0.5 * step / 10.0
    outputs = -1**step * jnp.ones((1, 1)) / 10 # * 3 * step / 10.0# reward
    # --------- ngclearn ---------
    clamp_weights(initial_ngclearn_weights)
    clamp_rewards(outputs)
    clamp_inputs(inputs)
    np.random.seed(43)
    ctx.adapt(t=1., dt=dt)
    print(f"[ngclearn] objective: {a.objective.value}")
    print(f"[ngclearn] weights: {a.weights.value}")
    print(f"[ngclearn] dWeights: {a.dWeights.value}")
    print(f"[ngclearn] step_count: {a.step_count.value}")
    print(f"[ngclearn] accumulated_gradients: {a.accumulated_gradients.value}")
    # -------- Expectation ---------
    print("--------------")
    np.random.seed(43)
    expected_objective, expected_grads = grad_fn(
        (expected_weights_mu, expected_weights_logstd),
        inputs,
        outputs,
    )
    # NOTE: Viet: negate the gradient because gradient in ngc-learn
    #   is gradient ascent, while gradient in JAX is gradient descent
    expected_grads = -jnp.concatenate([expected_grads[0], expected_grads[1]], axis=-1)
    expected_gradient_list.append(expected_grads)
    print(f"[Expectation] expected_weights: {expected_weights}")
    print(f"[Expectation] dWeights: {expected_grads}")
    print(f"[Expectation] objective: {expected_objective}")
    np.testing.assert_allclose(
        a.dWeights.value[0],
        expected_grads,
        atol=1e-8
    )
    np.testing.assert_allclose(
        a.objective.value,
        expected_objective,
        atol=1e-8
    )
    print()

    # Finally, check if the accumulated gradients are correct
    decay_list = jnp.asarray([decay**1, decay**0])
    expected_accumulated_gradients = jnp.mean(jnp.stack(expected_gradient_list, 0) * decay_list[:, None, None], axis=0)
    np.testing.assert_allclose(
        a.accumulated_gradients.value[0],
        expected_accumulated_gradients,
        atol=1e-8
    )


test_REINFORCESynapse1()

