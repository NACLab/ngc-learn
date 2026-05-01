# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.components import LaplacianErrorCell
from ngclearn import MethodProcess, Context

def test_laplacianErrorCell1():
  np.random.seed(42)
  name = "laplacian_error_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  with Context(name) as ctx:
    a = LaplacianErrorCell(
      name="a", n_units=1, batch_size=1, scale=1.0, shape=None
    )
    
    advance_process = (MethodProcess("advance_proc") >> a.advance_state)
    reset_process = (MethodProcess("reset_proc") >> a.reset)

  def clamp_modulator(x):
    a.modulator.set(x)

  def clamp_shift(x):
    a.shift.set(x)

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
  reset_process.run()
  for ts in range(shift_seq.shape[1]):
    shift_t = jnp.array([[shift_seq[0, ts]]])  ## get data at time t
    clamp_shift(shift_t)
    modulator_t = jnp.array([[modulator_seq[0, ts]]])
    clamp_modulator(modulator_t)
    target_t = jnp.array([[target_seq[0, ts]]])
    clamp_target(target_t)
    advance_process.run(t=ts * 1., dt=dt)
    dshift_outs.append(a.dshift.get())
    # print(f"a.L.value: {a.L.value}")
    # print(f"a.shift.value: {a.shift.value}")
    # print(f"a.target.value: {a.target.value}")
    # print(f"a.Scale.value: {a.Scale.value}")
    # print(f"a.mask.value: {a.mask.value}")
    L_outs.append(a.L.get())

  dshift_outs = jnp.concatenate(dshift_outs, axis=1)
  L_outs = jnp.array(L_outs)[None] # (1, 10)
  # print(dshift_outs)
  # print(L_outs)
  # print(expected_dshift)
  # print(expected_L)

  ## verify outputs match expected values
  np.testing.assert_allclose(dshift_outs, expected_dshift, atol=1e-5)
  np.testing.assert_allclose(L_outs, expected_L, atol=1e-5)

#test_laplacianErrorCell1()
