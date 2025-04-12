# %%

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import BernoulliErrorCell
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch


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
    advance_process = (Process("advance_proc") >> a.advance_state)
    ctx.wrap_and_add_command(jit(advance_process.pure), name="run")
    reset_process = (Process("reset_proc") >> a.reset)
    ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")

    # reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
    # ctx.add_command(wrap_command(jit(ctx.reset)), name="reset")
    # advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
    # ctx.add_command(wrap_command(jit(ctx.advance_state)), name="run")

    @Context.dynamicCommand
    def clamp(x):
      a.p.set(x)

    @Context.dynamicCommand
    def clamp_target(x):
      a.target.set(x)

  ## input spike train
  x_seq = jnp.asarray(np.random.randn(1, 10))
  target_seq = (jnp.arange(10)[None] - 5.0) / 2.0
  ## desired output/epsp pulses
  y_seq = jnp.asarray([[-2.8193381, -4976.9263, -2.1224928, -2939.0425, -1233.3916, -0.24662945, -708.30042, 0.28213939, 3550.8477, 1.3651246]], dtype=jnp.float32)

  outs = []
  ctx.reset()
  for ts in range(x_seq.shape[1]):
      x_t = jnp.array([[x_seq[0, ts]]])  ## get data at time t
      ctx.clamp(x_t)
      target_xt = jnp.array([[target_seq[0, ts]]])
      ctx.clamp_target(target_xt)
      ctx.run(t=ts * 1., dt=dt)
      outs.append(a.dp.value)
  outs = jnp.concatenate(outs, axis=1)
  # print(outs)
  ## output should equal input
  np.testing.assert_allclose(outs, y_seq, atol=1e-3)

# test_bernoulliErrorCell()
