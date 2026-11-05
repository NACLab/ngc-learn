# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components import HebbianAttentionPatchedSynapse
from numpy.testing import assert_array_equal

from ngclearn import MethodProcess, Context

def test_hebbianAttentionPatchedSynapse():
  np.random.seed(42)
  name = "hebbian_attention_patched_synapse_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms

  # model hyper
  shape = (10, 10)
  n_patch = 2
  batch_size = 1
  resist_scale = 1.0

  with Context(name) as ctx:
    a = HebbianAttentionPatchedSynapse(
      name="a",
      shape=shape,
      n_patch=n_patch,
      resist_scale=resist_scale,
      batch_size=batch_size
    )

    advance_process = (MethodProcess("advance_proc") >> a.advance_state)
    reset_process = (MethodProcess("reset_proc") >> a.reset)
    evolve_process = (MethodProcess("evolve_proc") >> a.evolve)

    def clamp_inputs(x):
      a.inputs.set(x)

    def clamp_pre(x):
      a.pre.set(x)

    def clamp_post(x):
      a.post.set(x)

  a.weights.set(jnp.ones((5, 15)) * 0.5)

  in_pre = jnp.ones((10, 10)) * 1.0
  in_post = jnp.ones((10, 10)) * 0.75

  reset_process.run()
  clamp_pre(in_pre)
  clamp_post(in_post)
  advance_process.run(t=1. * dt, dt=dt)
  evolve_process.run(t=1. * dt, dt=dt)

  print(a.weights.get())

  # Basic assertions to check learning dynamics
  assert a.weights.get().shape == (5, 15), ""
  assert a.weights.get()[0, 0] == 0.5, ""


# test_hebbianAttentionPatchedSynapse()


