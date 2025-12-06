# %%

import jax
from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components.synapses.modulated.REINFORCESynapse import REINFORCESynapse, _gaussian_logpdf
from numpy.testing import assert_array_equal

from ngclearn import Context, MethodProcess

import jax
import jax.numpy as jnp

def test_REINFORCESynapse1():
    # Testing reinforce synapse with learning stddev
    name = "reinforce_ctx"
    ## create seeding keys
    np.random.seed(42)
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    decay = 0.99
    initial_seed = 42
    mu_out_min = -jnp.inf
    mu_out_max = jnp.inf

    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = REINFORCESynapse(
            name="a", shape=(1,1), decay=decay,
            act_fx="tanh", key=subkeys[0], seed=initial_seed,
            mu_act_fx="tanh", mu_out_min=mu_out_min, mu_out_max=mu_out_max,
            scalar_stddev=-1.0
        )

        evolve_process = (MethodProcess("evolve_proc") >> a.evolve)
        reset_process = (MethodProcess("reset_proc") >> a.reset)

        def clamp_inputs(x):
            a.inputs.set(x)

        def clamp_rewards(x):
            assert x.ndim == 1, "Rewards must be a 1D array"
            a.rewards.set(x)

        def clamp_weights(x):
            a.weights.set(x)

    # Function definition
    _act = jax.nn.tanh
    def fn(params: dict, inputs: jax.Array, outputs: jax.Array, seed: jax.Array):
        W_mu, W_logstd = params
        activation = _act(inputs)
        mean = activation @ W_mu
        mean = jax.nn.tanh(mean)
        logstd = activation @ W_logstd
        std = jnp.exp(logstd.clip(-10.0, 2.0))
        sample = jax.random.normal(seed, mean.shape) * std + mean
        sample = jnp.clip(sample, mu_out_min, mu_out_max)
        logp = _gaussian_logpdf(jax.lax.stop_gradient(sample), mean, std).sum(-1)
        return (-logp * outputs).mean() * 1e-2
    grad_fn = jax.value_and_grad(fn)

    # Some setups
    expected_seed = jax.random.PRNGKey(initial_seed)
    expected_weights_mu = jnp.asarray([[0.13]])
    expected_weights_logstd = jnp.asarray([[0.04]])
    expected_weights = jnp.concatenate([expected_weights_mu, expected_weights_logstd], axis=-1)
    initial_ngclearn_weights = jnp.concatenate([expected_weights_mu, expected_weights_logstd], axis=-1)[None]
    expected_gradient_list = []
    reset_process.run()

    # Loop through 3 steps
    for step in range(10):
        expected_seed, expected_subseed = jax.random.split(expected_seed)

        # ---------------- Step {step} --------------------
        print(f"------------ [Step {step}] ------------")
        inputs = -1**step * jnp.ones((1, 1)) / 10  # * 0.5 * step / 10.0
        outputs = -1**step * jnp.ones((1,)) / 10 # * 3 * step / 10.0# reward
        # --------- ngclearn ---------
        clamp_weights(initial_ngclearn_weights)
        clamp_rewards(outputs)
        clamp_inputs(inputs)
        evolve_process.run(t=1., dt=dt)
        print(f"[ngclearn] objective: {a.objective.get()}")
        print(f"[ngclearn] weights: {a.weights.get()}")
        print(f"[ngclearn] dWeights: {a.dWeights.get()}")
        print(f"[ngclearn] step_count: {a.step_count.get()}")
        print(f"[ngclearn] accumulated_gradients: {a.accumulated_gradients.get()}")
        # -------- Expectation ---------
        print("--------------")
        expected_objective, expected_grads = grad_fn(
            (expected_weights_mu, expected_weights_logstd),
            inputs,
            outputs,
            expected_subseed
        )
        # NOTE: Viet: negate the gradient because gradient in ngc-learn
        #   is gradient ascent, while gradient in JAX is gradient descent
        expected_grads = -jnp.concatenate([expected_grads[0], expected_grads[1]], axis=-1)
        expected_gradient_list.append(expected_grads)
        print(f"[Expectation] expected_weights: {expected_weights}")
        print(f"[Expectation] dWeights: {expected_grads}")
        print(f"[Expectation] objective: {expected_objective}")
        np.testing.assert_allclose(
            a.dWeights.get()[0],
            expected_grads,
            atol=1e-8
        )
        np.testing.assert_allclose(
            a.objective.get(),
            expected_objective,
            atol=1e-8
        )
        print()

    # Finally, check if the accumulated gradients are correct
    decay_list = jnp.asarray([decay**i for i in range(len(expected_gradient_list))])[::-1]
    expected_accumulated_gradients = jnp.mean(jnp.stack(expected_gradient_list, 0) * decay_list[:, None, None], axis=0)
    np.testing.assert_allclose(
        a.accumulated_gradients.get()[0],
        expected_accumulated_gradients,
        atol=1e-9
    )


# test_REINFORCESynapse1()


def test_REINFORCESynapse2():
    # Testing reinforce synapse with scalar stddev = 2.0
    name = "reinforce_ctx2"
    ## create seeding keys
    np.random.seed(42)
    dkey = random.PRNGKey(1234)
    dkey, *subkeys = random.split(dkey, 6)
    dt = 1.  # ms
    decay = 0.99
    initial_seed = 42
    mu_out_min = -jnp.inf
    mu_out_max = jnp.inf
    scalar_stddev = 2.0

    # ---- build a simple Poisson cell system ----
    with Context(name) as ctx:
        a = REINFORCESynapse(
            name="a", shape=(1,1), decay=decay,
            act_fx="tanh", key=subkeys[0], seed=initial_seed,
            mu_act_fx="tanh", mu_out_min=mu_out_min, mu_out_max=mu_out_max,
            scalar_stddev=scalar_stddev
        )

        evolve_process = (MethodProcess("evolve_proc") >> a.evolve)
        reset_process = (MethodProcess("reset_proc") >> a.reset)

        def clamp_inputs(x):
            a.inputs.set(x)

        def clamp_rewards(x):
            assert x.ndim == 1, "Rewards must be a 1D array"
            a.rewards.set(x)

        def clamp_weights(x):
            a.weights.set(x)

    # Function definition
    _act = jax.nn.tanh
    def fn(params: dict, inputs: jax.Array, outputs: jax.Array, seed: jax.Array):
        W_mu, W_logstd = params
        activation = _act(inputs)
        mean = activation @ W_mu
        mean = jax.nn.tanh(mean)
        # logstd = activation @ W_logstd
        # std = jnp.exp(logstd.clip(-10.0, 2.0))
        std = scalar_stddev
        sample = jax.random.normal(seed, mean.shape) * std + mean
        sample = jnp.clip(sample, mu_out_min, mu_out_max)
        logp = _gaussian_logpdf(jax.lax.stop_gradient(sample), mean, std).sum(-1)
        return (-logp * outputs).mean() * 1e-2
    grad_fn = jax.value_and_grad(fn)

    # Some setups
    expected_seed = jax.random.PRNGKey(initial_seed)
    expected_weights_mu = jnp.asarray([[0.13]])
    expected_weights_logstd = jnp.asarray([[0.04]])
    expected_weights = jnp.concatenate([expected_weights_mu, expected_weights_logstd], axis=-1)
    initial_ngclearn_weights = jnp.concatenate([expected_weights_mu, expected_weights_logstd], axis=-1)[None]
    expected_gradient_list = []
    reset_process.run()

    # Loop through 3 steps
    for step in range(10):
        expected_seed, expected_subseed = jax.random.split(expected_seed)

        # ---------------- Step {step} --------------------
        print(f"------------ [Step {step}] ------------")
        inputs = -1**step * jnp.ones((1, 1)) / 10  # * 0.5 * step / 10.0
        outputs = -1**step * jnp.ones((1,)) / 10 # * 3 * step / 10.0# reward
        # --------- ngclearn ---------
        clamp_weights(initial_ngclearn_weights)
        clamp_rewards(outputs)
        clamp_inputs(inputs)
        evolve_process.run(t=1., dt=dt)
        print(f"[ngclearn] objective: {a.objective.get()}")
        print(f"[ngclearn] weights: {a.weights.get()}")
        print(f"[ngclearn] dWeights: {a.dWeights.get()}")
        print(f"[ngclearn] step_count: {a.step_count.get()}")
        print(f"[ngclearn] accumulated_gradients: {a.accumulated_gradients.get()}")
        # -------- Expectation ---------
        print("--------------")
        expected_objective, expected_grads = grad_fn(
            (expected_weights_mu, expected_weights_logstd),
            inputs,
            outputs,
            expected_subseed
        )
        # NOTE: Viet: negate the gradient because gradient in ngc-learn
        #   is gradient ascent, while gradient in JAX is gradient descent
        expected_grads = -jnp.concatenate([expected_grads[0], expected_grads[1]], axis=-1)
        expected_gradient_list.append(expected_grads)
        print(f"[Expectation] expected_weights: {expected_weights}")
        print(f"[Expectation] dWeights: {expected_grads}")
        print(f"[Expectation] objective: {expected_objective}")
        np.testing.assert_allclose(
            a.dWeights.get()[0],
            expected_grads,
            atol=1e-8
        )
        np.testing.assert_allclose(
            a.objective.get(),
            expected_objective,
            atol=1e-8
        )
        print()

    # Finally, check if the accumulated gradients are correct
    decay_list = jnp.asarray([decay**i for i in range(len(expected_gradient_list))])[::-1]
    expected_accumulated_gradients = jnp.mean(jnp.stack(expected_gradient_list, 0) * decay_list[:, None, None], axis=0)
    np.testing.assert_allclose(
        a.accumulated_gradients.get()[0],
        expected_accumulated_gradients,
        atol=1e-9
    )

# test_REINFORCESynapse2()

