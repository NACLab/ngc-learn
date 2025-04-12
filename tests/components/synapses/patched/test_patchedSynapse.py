# %%

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import PatchedSynapse
from ngcsimlib.compilers import compile_command, wrap_command
from numpy.testing import assert_array_equal

from ngcsimlib.compilers.process import Process, transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.context import Context
from ngcsimlib.utils.compartment import Get_Compartment_Batch


def test_patchedSynapse():
  np.random.seed(42)
  name = "patched_synapse_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  # model hyper
  shape = (10, 5)
  n_sub_models = 2
  stride_shape = (1, 1)
  batch_size = 1
  resist_scale = 1.0
  with Context(name) as ctx:
    a = PatchedSynapse(
      name="a",
      shape=shape,
      n_sub_models=n_sub_models,
      stride_shape=stride_shape,
      resist_scale=resist_scale,
      batch_size=batch_size,
      weight_init={"dist": "gaussian", "std": 0.1},
      bias_init={"dist": "constant", "value": 0.0}
    )

    advance_process = (Process("advance_proc") >> a.advance_state)
    ctx.wrap_and_add_command(jit(advance_process.pure), name="run")
    reset_process = (Process("reset_proc") >> a.reset)
    ctx.wrap_and_add_command(jit(reset_process.pure), name="reset")

    # Compile and add commands
    # reset_cmd, reset_args = ctx.compile_by_key(a, compile_key="reset")
    # ctx.add_command(wrap_command(jit(reset_cmd)), name="reset")
    # advance_cmd, advance_args = ctx.compile_by_key(a, compile_key="advance_state")
    # ctx.add_command(wrap_command(jit(advance_cmd)), name="run")

    @Context.dynamicCommand
    def clamp_inputs(x):
      a.inputs.set(x)

  inputs_seq = jnp.asarray(np.random.randn(1, 12))
  weights = a.weights.value
  biases = a.biases.value
  expected_outputs = (jnp.matmul(inputs_seq, weights) * resist_scale) + biases
  outputs_outs = []
  ctx.reset()
  ctx.clamp_inputs(inputs_seq)
  ctx.run(t=0., dt=dt)
  outputs_outs.append(a.outputs.value)
  outputs_outs = jnp.concatenate(outputs_outs, axis=1)
  # Verify outputs match expected values
  np.testing.assert_allclose(outputs_outs, expected_outputs, atol=1e-5)

