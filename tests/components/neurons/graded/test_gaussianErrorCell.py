# %%

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import GaussianErrorCell
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch


def test_gaussianErrorCell():
  np.random.seed(42)
  name = "gaussian_error_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  with Context(name) as ctx:
    a = GaussianErrorCell(
      name="a", n_units=1, batch_size=1, sigma=1.0, shape=None
    )
    advance_process = (Process("advance_proc") >> a.advance_state)
    ctx.wrap_and_add_command(jit(advance_process.pure), name="run")
    reset_process = (Process("reset_proc") >> a.reset)
    ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")

    # reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
    # ctx.add_command(wrap_command(jit(ctx.reset)), name="reset")
    # advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
    # ctx.add_command(wrap_command(jit(ctx.advance_state)), name="run")

    @Context.dynamicCommand
    def clamp_mu(x):
      a.mu.set(x)

    @Context.dynamicCommand
    def clamp_target(x):
      a.target.set(x)

  ## input sequence
  mu_seq = jnp.asarray(np.random.randn(1, 10))
  target_seq = (jnp.arange(10)[None] - 5.0) / 2.0
  ## expected output based on the Gaussian error cell formula
  ## L = -0.5 * (target - mu)^2 / sigma, dmu = (target - mu) / sigma
  expected_dmu = (target_seq - mu_seq) / 1.0  # sigma = 1.0
  expected_L = -0.5 * jnp.square(target_seq - mu_seq) / 1.0

  dmu_outs = []
  L_outs = []
  ctx.reset()
  for ts in range(mu_seq.shape[1]):
      mu_t = jnp.array([[mu_seq[0, ts]]])  ## get data at time t
      ctx.clamp_mu(mu_t)
      target_t = jnp.array([[target_seq[0, ts]]])
      ctx.clamp_target(target_t)
      ctx.run(t=ts * 1., dt=dt)
      dmu_outs.append(a.dmu.value)
      L_outs.append(a.L.value)

  dmu_outs = jnp.concatenate(dmu_outs, axis=1)
  L_outs = jnp.array(L_outs)[None] # (1, 10)
  # print(dmu_outs.shape)
  # print(L_outs.shape)
  # print(expected_dmu.shape)
  # print(expected_L.shape)

  ## verify outputs match expected values
  np.testing.assert_allclose(dmu_outs, expected_dmu, atol=1e-5)
  np.testing.assert_allclose(L_outs, expected_L, atol=1e-5)

# test_gaussianErrorCell()