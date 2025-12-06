# %%

from jax import numpy as jnp, random, jit
import numpy as np
np.random.seed(42)
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components import PatchedSynapse

from ngclearn import MethodProcess, Context



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
      weight_init=dist.gaussian(std=0.1), #{"dist": "gaussian", "std": 0.1},
      bias_init=dist.constant(value=0.) #{"dist": "constant", "value": 0.0}
    )

    advance_process = (MethodProcess("advance_proc") >> a.advance_state)
    reset_process = (MethodProcess("reset_proc") >> a.reset)

    def clamp_inputs(x):
      a.inputs.set(x)

  inputs_seq = jnp.asarray(np.random.randn(1, 12))
  weights = a.weights.get()
  biases = a.biases.get()
  expected_outputs = (jnp.matmul(inputs_seq, weights) * resist_scale) + biases
  outputs_outs = []
  reset_process.run()
  clamp_inputs(inputs_seq)
  advance_process.run(t=0., dt=dt)
  outputs_outs.append(a.outputs.get())
  outputs_outs = jnp.concatenate(outputs_outs, axis=1)
  # Verify outputs match expected values
  np.testing.assert_allclose(outputs_outs, expected_outputs, atol=1e-5)


test_patchedSynapse()

