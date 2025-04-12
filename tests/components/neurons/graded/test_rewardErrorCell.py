# %%

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import RewardErrorCell
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch


def test_rewardErrorCell():
  np.random.seed(42)
  name = "reward_error_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  alpha = 0.1  # decay factor for moving average
  with Context(name) as ctx:
    a = RewardErrorCell(
      name="a", n_units=1, alpha=alpha, ema_window_len=10, 
      use_online_predictor=True, batch_size=1
    )
    advance_process = (Process("advance_proc") >> a.advance_state)
    ctx.wrap_and_add_command(jit(advance_process.pure), name="run")
    reset_process = (Process("reset_proc") >> a.reset)
    ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
    evolve_process = (Process("evolve_proc") >> a.evolve)
    ctx.wrap_and_add_command(jit(evolve_process.pure), name="evolve")

    # reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
    # ctx.add_command(wrap_command(jit(ctx.reset)), name="reset")
    # advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
    # ctx.add_command(wrap_command(jit(ctx.advance_state)), name="run")
    # evolve_cmd, evolve_args = ctx.compile_by_key(a, compile_key="evolve")
    # ctx.add_command(wrap_command(jit(ctx.evolve)), name="evolve")

    @Context.dynamicCommand
    def clamp_reward(x):
      a.reward.set(x)

  ## input reward sequence
  reward_seq = jnp.array([[1.0, 0.5, 0.0, 2.0, 1.5, 0.0, 1.0, 0.5, 0.0, 1.0]])

  # NOTE: expected outputs: look at each function in the cell: e.g., advance_state, evolve, reset, to test
  # rpe = reward - mu, mu = mu * (1 - alpha) + reward * alpha
  # These expectation numbers will be computed in the loop below
  expected_mu = np.zeros((1, 10))
  expected_rpe = np.zeros((1, 10))
  expected_accum_reward = np.zeros((1, 10))
  # Calculate expected values
  mu_t = 0.0
  accum_t = 0.0
  for t in range(10):
    reward_t = reward_seq[0, t]
    # print(f"reward_t: {reward_t}")
    accum_t += reward_t
    # print(f"accum_t: {accum_t}")
    expected_accum_reward[0, t] = np.asarray(accum_t) # NOTE: Formula: accum_reward = accum_reward + reward
    expected_rpe[0, t] = np.asarray(reward_t - mu_t) # NOTE: Formula: rpe = reward - mu
    mu_t = mu_t * (1 - alpha) + reward_t * alpha  # NOTE: Formula: mu = mu * (1. - alpha) + reward * alpha
    # print(f"mu_t: {mu_t}")
    expected_mu[0, t] = np.asarray(mu_t)

  mu_outs = []
  rpe_outs = []
  accum_reward_outs = []
  ctx.reset()
  for ts in range(reward_seq.shape[1]):
      reward_t = jnp.array([[reward_seq[0, ts]]])  ## get reward at time t
      ctx.clamp_reward(reward_t)
      ctx.run(t=ts * 1., dt=dt)
      mu_outs.append(a.mu.value)
      rpe_outs.append(a.rpe.value)
      accum_reward_outs.append(a.accum_reward.value)

  # Test evolve function
  ctx.evolve(t=10 * 1., dt=dt)
  final_mu = a.mu.value
  # print(f"final_mu: {final_mu}")

  mu_outs = jnp.concatenate(mu_outs, axis=1)
  # print(mu_outs)
  rpe_outs = jnp.concatenate(rpe_outs, axis=1)
  # print(rpe_outs)
  accum_reward_outs = jnp.concatenate(accum_reward_outs, axis=1)
  # print(accum_reward_outs)

  ## verify outputs match expected values
  np.testing.assert_allclose(mu_outs, expected_mu, atol=1e-5)
  np.testing.assert_allclose(rpe_outs, expected_rpe, atol=1e-5)
  np.testing.assert_allclose(accum_reward_outs, expected_accum_reward, atol=1e-5)

  # Verify final mu after evolve
  # Basically copy the formula from the evolve function: r = accum_reward/n_ep_steps
  # and this one as well: `mu = (1. - 1./ema_window_len) * mu + (1./ema_window_len) * r`
  expected_final_mu = (1 - 1/10) * mu_outs[0, -1] + (1/10) * (accum_reward_outs[0, -1] / 10)
  np.testing.assert_allclose(final_mu, expected_final_mu, atol=1e-5)

# test_rewardErrorCell()