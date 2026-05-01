# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components.neurons.graded.rateCell import RateCell
from numpy.testing import assert_array_equal

from ngclearn import Context, MethodProcess


def test_RateCell1():
  name = "rate_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  with Context(name) as ctx:
    a = RateCell(
      name="a", n_units=1, tau_m=50., prior=("gaussian", 0.), act_fx="identity",
      threshold=("none", 0.), integration_type="euler",
      batch_size=1, resist_scale=1., shape=None, is_stateful=True
    )
    advance_process = (MethodProcess("advance_proc") >> a.advance_state)
    reset_process = (MethodProcess("reset_proc") >> a.reset)

  def clamp(x):
    a.j.set(x)

  ## input spike train
  x_seq = jnp.ones((1, 10))
  ## desired output/epsp pulses
  y_seq = jnp.asarray([[0.02, 0.04, 0.06, 0.08, 0.09999999999999999, 0.11999999999999998, 0.13999999999999999, 0.15999999999999998, 0.17999999999999998, 0.19999999999999998]], dtype=jnp.float32)

  outs = []
  reset_process.run()
  for ts in range(x_seq.shape[1]):
      x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
      clamp(x_t)
      advance_process.run(t=ts * 1., dt=dt)
      outs.append(a.z.get())
  outs = jnp.concatenate(outs, axis=1)
  # print(outs)
  ## output should equal input
  # assert_array_equal(outs, y_seq, tol=1e-3)
  np.testing.assert_allclose(outs, y_seq, atol=1e-3)

#test_RateCell1()
