# Lesson 3: Evolving Synaptic Efficacies

In this tutorial, we will extend a controller with three components,
two cell components connected with a synaptic cable component, to incorporate a
basic a two-factor Hebbian adjustment process.

## Adding a Learnable Synapses to a Multi-Component System

Let us start by building a controller similar to previous lessons with the one
exception that now we will trigger the synaptic connection between `a` and `b`
to adapt via a simple 2-factor Hebbian rule. This Hebbian rule will require us
to wire the output compartment of `a` to the pre-synaptic compartment of the
synapse `Wab` and the output compartment of `b` to the post-synaptic
compartment of `Wab`. This will wire in the two relevant factors needed to
compute a simple Hebbian adjustment.

We do this specifically as follows:

```python
from ngcsimlib.controller import Controller
from jax import numpy as jnp, random, nn, jit

## create seeding keys
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

## create simple dynamical system: a --> w_ab --> b
model = Controller()
a = model.add_component("rate", name="a", n_units=1, tau_m=0.,
                        act_fx="identity", leakRate=0., key=subkeys[0])
b = model.add_component("rate", name="b", n_units=1, tau_m=0.,
                        act_fx="identity", leakRate=0., key=subkeys[1])
Wab = model.add_component("hebbian", name="Wab", shape=(1, 1),
                          eta=1., wInit=("constant", 1., None), w_bound=0.,
                          key=subkeys[3])
## wire a to w_ab and wire w_ab to b
model.connect(a.name, a.outputCompartmentName(), Wab.name, Wab.inputCompartmentName())
model.connect(Wab.name, Wab.outputCompartmentName(), b.name, b.inputCompartmentName())

model.connect(a.name, a.outputCompartmentName(), Wab.name, Wab.presynapticCompartmentName())
model.connect(b.name, b.outputCompartmentName(), Wab.name, Wab.postsynapticCompartmentName())

## configure desired commands for simulation object
model.add_command("reset", command_name="reset",
                  component_names=[a.name, Wab.name, b.name],
                  reset_name="do_reset")
model.add_command(
    "advance", command_name="advance",
    component_names=[a.name, Wab.name, b.name]
)
model.add_command("evolve", command_name="evolve", component_names=[Wab.name])
model.add_command("clamp", command_name="clamp_data",
                  component_names=[a.name], compartment=a.inputCompartmentName(),
                  clamp_name="x")
## pin the commands to the object
model.add_step("advance")
model.add_step("evolve")
```

Now with our simple system above created, we will now run a simple sequence
of one-dimensional "spike" data through it and evolve the synapse every time
step like so:

```python
## run some data through the dynamical system
x_seq = jnp.asarray([[1, 1, 0, 0, 1]], dtype=jnp.float32)

model.reset(do_reset=True)
print("{}: Wab = {}".format(-1, model.components["Wab"].weights))
for ts in range(x_seq.shape[1]):
  x_t = jnp.expand_dims(x_seq[0,ts], axis=0) ## get data at time t
  model.clamp_data(x=x_t)
  model.runCycle(t=ts*1., dt=1.)
  print(" {}: input = {} ~> Wab = {}".format(ts, x_t, model.components["Wab"].weights))
```

Your code should produce the same output (towards the bottom):

```console
-1: Wab = [[1.]]
 0: input = [1.] ~> Wab = [[2.]]
 1: input = [1.] ~> Wab = [[4.]]
 2: input = [0.] ~> Wab = [[4.]]
 3: input = [0.] ~> Wab = [[4.]]
 4: input = [1.] ~> Wab = [[8.]]
```

Notice that for every non-spike (a value of `0`), the synaptic value remains
the same (because the product of a pre-synaptic value of `0` with a post-synaptic
value of anything -- in this case, also a `0` -- is simply `0`, meaning no
change will be applied to the synapse). For every spike (a value of `1`), we
get a synaptic change equal to `dW = input * (Wab * input)`; so for the
first time-step, the weight will change according to
`W = W + eta * dW = W + dW` and `dW = 1 * (1 * 1) = 1`, whereas, for the
second time-step, `W` will be increased by `dW = 1 * (2 * 1) = 2` (yielding a
  new synaptic strength of `W = 4`).

You have now created your first plastic, evolving neuronal system.
