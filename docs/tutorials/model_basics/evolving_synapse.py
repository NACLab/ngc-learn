# %%

from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngcsimlib.commands import Command

from ngclearn.components import HebbianSynapse, RateCell

from jax import numpy as jnp, random, jit
import numpy as np

## create seeding keys
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

## create simple system with only one F-N cell
with Context("Circuit") as circuit:
  a = RateCell(name="a", n_units=1, tau_m=0.,
    act_fx="identity", key=subkeys[0])
  b = RateCell(name="b", n_units=1, tau_m=0.,
    act_fx="identity", key=subkeys[1])

  Wab = HebbianSynapse(name="Wab", shape=(1, 1), eta=1.,
    signVal=-1., wInit=("constant", 1., None),
    w_bound=0., key=subkeys[3])

  # wire output compartment (rate-coded output zF) of RateCell `a` to input compartment of HebbianSynapse `Wab`
  Wab.inputs << a.zF
  # wire output compartment of HebbianSynapse `Wab` to input compartment (electrical current j) RateCell `b`
  b.j << Wab.outputs

  # wire output compartment (rate-coded output zF) of RateCell `a` to presynaptic compartment of HebbianSynapse `Wab`
  Wab.pre << a.zF
  # wire output compartment (rate-coded output zF) of RateCell `b` to postsynaptic compartment of HebbianSynapse `Wab`
  Wab.post << b.zF

  ## create and compile core simulation commands
  reset_cmd, reset_args = circuit.compile_command_key(a, Wab, b, compile_key="reset")
  circuit.add_command(wrap_command(jit(circuit.reset)), name="reset")

  advance_cmd, advance_args = circuit.compile_command_key(a, Wab, b, compile_key="advance_state")
  circuit.add_command(wrap_command(jit(circuit.advance_state)), name="advance")

  evolve_cmd, evolve_args = circuit.compile_command_key(Wab, compile_key="evolve")
  circuit.add_command(wrap_command(jit(circuit.evolve)), name="evolve")

  ## set up non-compiled utility commands
  @Context.dynamicCommand
  def clamp(x):
    a.j.set(x)


## run some data through the dynamical system
x_seq = jnp.asarray([[1, 1, 0, 0, 1]], dtype=jnp.float32)

circuit.reset()
print("{}: Wab = {}".format(-1, Wab.weights.value))
for ts in range(x_seq.shape[1]):
  x_t = jnp.expand_dims(x_seq[0,ts], axis=0) ## get data at time t
  circuit.clamp(x_t)
  circuit.advance(ts*1., 1.)
  circuit.evolve(ts*1., 1.)
  print(" {}: input = {} ~> Wab = {}".format(ts, x_t, Wab.weights.value))

