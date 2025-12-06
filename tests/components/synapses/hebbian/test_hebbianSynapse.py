# %%

from jax import numpy as jnp, random, jit

import numpy as np
np.random.seed(42)
from ngclearn.components.synapses.hebbian.hebbianSynapse import HebbianSynapse

from numpy.testing import assert_array_equal
from ngclearn import Context, MethodProcess


def test_hebbianSynapse():
  np.random.seed(42)
  name = "hebbian_synapse_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms

  # model hyper
  shape = (10, 5)
  batch_size = 1
  resist_scale = 1.0

  with Context(name) as ctx:
    a = HebbianSynapse(
      name="a",
      shape=shape,
      resist_scale=resist_scale,
      batch_size=batch_size,
      prior = ("gaussian", 0.01)
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

  # Test input sequence
  # Initial weights
  a.weights.set(jnp.ones((10, 5)) * 0.5) 

  in_pre = jnp.ones((1, 10)) * 1.0
  in_post = jnp.ones((1, 5)) * 0.75

  reset_process.run()
  clamp_pre(in_pre)
  clamp_post(in_post)
  advance_process.run(t=1. * dt, dt=dt)
  evolve_process.run(t=1. * dt, dt=dt)

  #print(a.weights.get())

  # Basic assertions to check learning dynamics
  assert a.weights.get().shape == (10, 5), ""
  assert a.weights.get()[0, 0] == 0.5, ""

#test_hebbianSynapse()

