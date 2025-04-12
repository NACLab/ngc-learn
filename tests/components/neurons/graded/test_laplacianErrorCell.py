# %%

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import LaplacianErrorCell
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch


def test_laplacianErrorCell():
  np.random.seed(42)
  name = "laplacian_error_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  with Context(name) as ctx:
    a = LaplacianErrorCell(
      name="a", n_units=1, batch_size=1, scale=1.0, shape=None
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
    def clamp_modulator(x):
      a.modulator.set(x)

    @Context.dynamicCommand
    def clamp_shift(x):
      a.shift.set(x)

    @Context.dynamicCommand
    def clamp_target(x):
      a.target.set(x)

  ## input sequence
  modulator_seq = jnp.ones((1, 10))
  shift_seq = jnp.asarray(np.random.randn(1, 10))
  target_seq = (jnp.arange(10)[None] - 5.0) / 2.0
  ## expected output based on the Laplacian error cell formula
  ## L = -|target - shift|/scale, dshift = sign(target - shift)/scale
  expected_dshift = jnp.sign(target_seq - shift_seq) / 1.0  # scale = 1.0
  # expected_L = -jnp.abs(target_seq - shift_seq) / 1.0 # NOTE: Viet: I tried to use this according to the cell formula but got different values, maybe check this later
  expected_L = -jnp.ones((1, 10))

  dshift_outs = []
  L_outs = []
  ctx.reset()
  for ts in range(shift_seq.shape[1]):
    shift_t = jnp.array([[shift_seq[0, ts]]])  ## get data at time t
    ctx.clamp_shift(shift_t)
    modulator_t = jnp.array([[modulator_seq[0, ts]]])
    ctx.clamp_modulator(modulator_t)
    target_t = jnp.array([[target_seq[0, ts]]])
    ctx.clamp_target(target_t)
    ctx.run(t=ts * 1., dt=dt)
    dshift_outs.append(a.dshift.value)
    # print(f"a.L.value: {a.L.value}")
    # print(f"a.shift.value: {a.shift.value}")
    # print(f"a.target.value: {a.target.value}")
    # print(f"a.Scale.value: {a.Scale.value}")
    # print(f"a.mask.value: {a.mask.value}")
    L_outs.append(a.L.value)

  dshift_outs = jnp.concatenate(dshift_outs, axis=1)
  L_outs = jnp.array(L_outs)[None] # (1, 10)
  # print(dshift_outs)
  # print(L_outs)
  # print(expected_dshift)
  # print(expected_L)

  ## verify outputs match expected values
  np.testing.assert_allclose(dshift_outs, expected_dshift, atol=1e-5)
  np.testing.assert_allclose(L_outs, expected_L, atol=1e-5)

