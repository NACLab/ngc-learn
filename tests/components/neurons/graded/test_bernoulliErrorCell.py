# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components import BernoulliErrorCell
from ngclearn import MethodProcess, Context

def test_bernoulliErrorCell():
  np.random.seed(42)
  name = "bernoulli_error_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  with Context(name) as ctx:
    a = BernoulliErrorCell(
      name="a", n_units=1, batch_size=1, input_logits=False, shape=None
    )
    advance_process = (MethodProcess("advance_proc") >> a.advance_state)
    reset_process = (MethodProcess("reset_proc") >> a.reset)

    def clamp(x):
      a.p.set(x)

    def clamp_target(x):
      a.target.set(x)

  ## input spike train
  x_seq = jnp.asarray(np.random.randn(1, 10))
  target_seq = (jnp.arange(10)[None] - 5.0) / 2.0
  ## desired output/epsp pulses
  y_seq = jnp.asarray([[-2.8193381, -4976.9263, -2.1224928, -2939.0425, -1233.3916, -0.24662945, -708.30042, 0.28213939, 3550.8477, 1.3651246]], dtype=jnp.float32)

  outs = []
  reset_process.run()
  for ts in range(x_seq.shape[1]):
      x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
      clamp(x_t)
      target_xt = jnp.array([[target_seq[0, ts]]])
      clamp_target(target_xt)
      advance_process.run(t=ts * 1., dt=dt)
      outs.append(a.dp.get())
  outs = jnp.concatenate(outs, axis=1)
  # print(outs)
  ## output should equal input
  np.testing.assert_allclose(outs, y_seq, atol=1e-3)

# test_bernoulliErrorCell()
