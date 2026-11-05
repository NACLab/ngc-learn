# %%

from jax import numpy as jnp, random, jit, nn
import numpy as np
np.random.seed(42)
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components import AttentionPatchedSynapse
from ngclearn import MethodProcess, Context


def test_attentionPatchedSynapse():
  np.random.seed(42)
  name = "attention_patched_synapse_ctx"
  dkey = random.PRNGKey(42)
  dkey, *subkeys = random.split(dkey, 100)
  dt = 1.  # ms
  # model hyper
  shape = (10, 10)
  n_patch = 2
  batch_size = 1
  resist_scale = 1.0
  with Context(name) as ctx:
    a = AttentionPatchedSynapse(
      name="a",
      shape=shape,
      n_patch=n_patch,
      resist_scale=resist_scale,
      batch_size=batch_size,
      weight_init=dist.gaussian(std=0.1),
      bias_init=dist.constant(value=0.)
    )

    advance_process = (MethodProcess("advance_proc") >> a.advance_state)
    reset_process = (MethodProcess("reset_proc") >> a.reset)
    
    def clamp_inputs(x):
      a.inputs.set(x)

  inputs_seq = jnp.asarray(np.random.randn(1, 10))
  weights = a.weights.get()
  biases = a.biases.get()
  zF = inputs_seq.reshape(n_patch, -1)
  q, k, v = jnp.split(jnp.matmul(zF, weights) + biases, 3, axis=1)
  A = nn.softmax(jnp.matmul(q, k.T) / jnp.sqrt(q.shape[1]), axis=-1)
  expected_outputs = jnp.matmul(A, v).reshape(1, -1) * resist_scale
  outputs_outs = []
  reset_process.run()
  clamp_inputs(inputs_seq)
  advance_process.run(t=0., dt=dt)
  outputs_outs.append(a.outputs.get())
  outputs_outs = jnp.concatenate(outputs_outs, axis=1)
  # Verify outputs match expected values
  np.testing.assert_allclose(outputs_outs, expected_outputs, atol=1e-5)


test_attentionPatchedSynapse()





