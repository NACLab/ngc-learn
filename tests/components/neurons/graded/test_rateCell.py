from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)
from ngclearn.components import RateCell
from ngcsimlib.compilers import compile_command, wrap_command

def test_rateCell1():
  ## create seeding keys
  dkey = random.PRNGKey(1234)
  dkey, *subkeys = random.split(dkey, 6)
  # in_dim = 9  # ... dimension of patch data ...
  # hid_dim = 9  # ... number of atoms in the dictionary matrix
  dt = 1.  # ms
  T = 300  # ms # (OR) number of E-steps to take during inference
  # ---- build a sparse coding linear generative model with a Cauchy prior ----
  with Context("Circuit") as circuit:
      a = RateCell(name="a", n_units=1, tau_m=0.,
                  act_fx="identity", key=subkeys[0])
      b = RateCell(name="b", n_units=1, tau_m=0.,
                  act_fx="identity", key=subkeys[1])

      # wire output compartment (rate-coded output zF) of RateCell `a` to input compartment of HebbianSynapse `Wab`

      # wire output compartment of HebbianSynapse `Wab` to input compartment (electrical current j) RateCell `b`
      b.j << a.zF

      ## create and compile core simulation commands
      reset_cmd, reset_args = circuit.compile_by_key(a, b, compile_key="reset")
      circuit.add_command(wrap_command(jit(circuit.reset)), name="reset")

      advance_cmd, advance_args = circuit.compile_by_key(a, b,
                                                          compile_key="advance_state")
      circuit.add_command(wrap_command(jit(circuit.advance_state)), name="advance")


      ## set up non-compiled utility commands
      @Context.dynamicCommand
      def clamp(x):
          a.j.set(x)

      x_seq = jnp.asarray([[1, 1, 0, 0, 1]], dtype=jnp.float32)

      circuit.reset()
      for ts in range(x_seq.shape[1]):
          x_t = jnp.expand_dims(x_seq[0,ts], axis=0) ## get data at time t
          circuit.clamp(x_t)
          circuit.advance(t=ts*1., dt=1.)

  print(a.zF.value)
  # assertion here if needed!
