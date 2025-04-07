# %%

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import HebbianSynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch


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

    advance_process = (Process("advance_proc") >> a.advance_state)
    ctx.wrap_and_add_command(jit(advance_process.pure), name="run")
    reset_process = (Process("reset_proc") >> a.reset)
    ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")
    evolve_process = (Process("evolve_proc") >> a.evolve)
    ctx.wrap_and_add_command(jit(evolve_process.pure), name="evolve")

    # Compile and add commands
    # reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
    # ctx.add_command(wrap_command(jit(reset_cmd)), name="reset")
    # advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
    # ctx.add_command(wrap_command(jit(advance_cmd)), name="run")
    # evolve_cmd, evolve_args = ctx.compile_by_key(a, compile_key="evolve")
    # ctx.add_command(wrap_command(jit(evolve_cmd)), name="evolve")

    @Context.dynamicCommand
    def clamp_inputs(x):
      a.inputs.set(x)

    @Context.dynamicCommand
    def clamp_pre(x):
      a.pre.set(x)

    @Context.dynamicCommand
    def clamp_post(x):
      a.post.set(x)

  # Test input sequence
  # Initial weights
  a.weights.set(jnp.ones((10, 5)) * 0.5) 

  in_pre = jnp.ones((1, 10)) * 1.0
  in_post = jnp.ones((1, 5)) * 0.75

  ctx.reset()
  clamp_pre(in_pre)
  clamp_post(in_post)
  ctx.run(t=1. * dt, dt=dt)
  ctx.evolve(t=1. * dt, dt=dt)

  print(a.weights.value)

  # Basic assertions to check learning dynamics
  assert a.weights.value.shape == (10, 5), ""
  assert a.weights.value[0, 0] == 0.5, ""

# test_hebbianSynapse() 