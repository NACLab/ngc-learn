# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components import GaussianErrorCell

from ngclearn import MethodProcess, Context


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
    advance_process = (MethodProcess("advance_proc") >> a.advance_state)
    reset_process = (MethodProcess("reset_proc") >> a.reset)

    def clamp_mu(x):
      a.mu.set(x)

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
  reset_process.run()
  for ts in range(mu_seq.shape[1]):
      mu_t = jnp.array([[mu_seq[0, ts]]])  ## get data at time t
      clamp_mu(mu_t)
      target_t = jnp.array([[target_seq[0, ts]]])
      clamp_target(target_t)
      advance_process.run(t=ts * 1., dt=dt)
      dmu_outs.append(a.dmu.get())
      L_outs.append(a.L.get())

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
