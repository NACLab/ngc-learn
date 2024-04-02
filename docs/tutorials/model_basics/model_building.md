# Lesson 2: Building a Model

In this tutorial, we will build a simple controller made up of three components:
two simple graded cells that are connected by one synaptic cable.

## Setting Up the Desired Components

First, to ensure that we pull out the right modeling pieces from the
ngc-learn/ngcsimlib "toolbox", we should write our project's JSON configuration
file as follows:

```json
[
  {
    "absolute_path": "ngcsimlib.commands",
    "attributes": [
      {
        "name": "AdvanceState",
        "keywords": ["advance"]
      },
      {
        "name": "Clamp",
        "keywords": ["clamp"]
      },
      {
        "name": "Reset",
        "keywords": ["reset"]
      }
    ]
  },
  {
    "absolute_path": "ngclearn.components",
    "attributes": [
      {"name": "RateCell",
        "keywords": ["rate"]
      },
      {
        "name": "HebbianSynapse",
        "keywords": ["hebbian"]
      }
    ]
  }
]
```

where we see above that we have chosen to pull out the
[RateCell](ngclearn.components.neurons.graded.rateCell) and
the [HebbianSynapse](ngclearn.components.synapses.hebbian.hebbianSynapse)
components along with the `AdvanceState` (keyword-bound to `advance`),
`Reset` (keyword-bound to `reset`), and `Clamp` (keyword-bound to `clamp`) commands.
The above JSON snippets would be placed in a JSON configuration file, usually
in the default local folder, e.g., in `/json_files/modules.json` (though one
can name this folder and config file anything they desire -- so long as the
proper flag is passed when executing the later Python script if different
names beside the default are used).

We will next arrange these together to craft our system with the parts selected
in our JSON configuration within a Python script.

## Instantiating the Dynamical System as a Controller

With the JSON configuration in place, we can now readily use the modeling
pieces that we felt were useful in building our simple 3-element system.
Concretely, we instantiate a simulation controller as well as the desired
components we would like to build into it:

```python
from ngcsimlib.controller import Controller
from jax import numpy as jnp, random

## create seeding keys (JAX-style)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 4)

## create simple dynamical system: a --> w_ab --> b
model = Controller() ## the simulation object
a = model.add_component("rate", name="a", n_units=1, tau_m=0.,
                        act_fx="identity", leakRate=0., key=subkeys[0])
b = model.add_component("rate", name="b", n_units=1, tau_m=20.,
                        act_fx="identity", leakRate=0., key=subkeys[1])
Wab = model.add_component("hebbian", name="Wab", shape=(1, 1),
                          wInit=("constant", 1., None), key=subkeys[2])
```

Next, we will want to wire together the three components we have embedded into
our controller, connecting `a` to node `b` through synaptic cable `Wab`. In
other words, this means that the output compartment of `a` must be wired to the
input compartment of transformation `Wab` and the output compartment of `Wab`
must be wired to the input compartment of `b`. In code, this is done as follows:

```python                        
## wire a to w_ab and wire w_ab to b
model.connect(a.name, a.outputCompartmentName(), Wab.name, Wab.inputCompartmentName())
model.connect(Wab.name, Wab.outputCompartmentName(), b.name, b.inputCompartmentName())
```

Finally, to make our dynamical system do something for each step of simulated
time, we must append a few basic commands
(see [Understanding Commands](../foundations/commands.md) to the controller.
The commands we will want, as implied by our JSON configuration that we put
together at the start of this tutorial, include a `reset` (which will
initialize the compartments within each node to their resting values,
i.e., generally zero, if they have them -- this will only end up affecting
nodes `a` and `b` since a basic synapse component like `Wab` does not have a
base/resting value), an `advance` (which moves all the nodes one step
forward in time according to their compartments' ODEs), and `clamp` (which will
allow us to insert data into particular nodes).
This is simply done with the following few lines:

```python
## configure desired commands for simulation object
model.add_command("reset", command_name="reset",
                  component_names=[a.name, Wab.name, b.name],
                  reset_name="do_reset")
model.add_command(
    "advance", command_name="advance",
    component_names=[a.name, Wab.name, b.name]
)
model.add_command("clamp", command_name="clamp_data",
                  component_names=[a.name], compartment=a.inputCompartmentName(),
                  clamp_name="x")
## pin the commands to the object
model.add_step("advance")
```

where we also notice we have one line that makes the controller aware that it
must execute the `advance` command for all components in its argument list
`component_names` one time within a simulation cycle (under
the hood, this lets the controller know that, for one cycle of computation
within the system, marching on from time step to the next involves calling
the `advance` command for cell `a`, then synapse `Wab`, and finally cell `b`).

## Running the Dynamical System's Controller

With our simple 3-component dynamical system built, we may now run it on a
simple sequence of one-dimensional real-valued numbers:

```python
## run some data through our simple dynamical system
x_seq = jnp.asarray([[1., 2., 3., 4., 5.]], dtype=jnp.float32)

model.reset(True)
for ts in range(x_seq.shape[1]):
  x_t = jnp.expand_dims(x_seq[0,ts], axis=0) ## get data at time ts
  model.clamp_data(x_t)
  model.runCycle(t=ts*1., dt=1.)
  ## naively extract simple statistics at time ts and print them to I/O
  a_out = model.components["a"].outputCompartment
  aName = model.components["a"].outputCompartmentName()
  b_out = model.components["b"].outputCompartment
  bName = model.components["b"].outputCompartmentName()
  print(" {}: a.{} = {} ~> b.{} = {}".format(ts, aName, a_out, bName, b_out))
```

and, assuming you place your code above in a Python script
(e.g., `run_lesson2.py`), we should obtain output in your terminal as below:

```console
$ python run_lesson2.py
 0: a.zF = [1.] ~> b.zF = [[0.05]]
 1: a.zF = [2.] ~> b.zF = [[0.15]]
 2: a.zF = [3.] ~> b.zF = [[0.3]]
 3: a.zF = [4.] ~> b.zF = [[0.5]]
 4: a.zF = [5.] ~> b.zF = [[0.75]]
```

The simple 3-component system simulated above merely transforms the input
sequence into another time-evolving series. For the curious, in your code above,
you modeled a very simple non-leaky integration of cell `b` injected with some
value produced by `a` (since `Wab = 1`, the synapses had no effect and merely
copies the value along). While node `a` is always clamped to a value as per the
clamp command call we constructed and call above (even though its time constant
was `tau_m = 0` ms, meaning that it reduces to a stateless "feedforward" cell),
b had a time constant you set to `tau_m = 20` ms. This means, as can be confirmed
by inspecting the API for `RateCell`, with your integration time constant
`dt = 1` ms:
1. at time step `ts = 0`, the value clamped to `a`, i.e., `1`, was multiplied by
   `1/20 = 0.05` and then added `b`'s internal state (which started at the value
   of `0` through the reset command called before the for-loop);
2. at step `ts = 1`, the value clamped to `a`, i.e., `2`, was multiplied by
   `0.05` (yielding `0.1`) and then added to `b`'s current state -- meaning that
   the new state becomes `0.05 + 0.1 = 0.15`;
3. at `ts = 2`, a value `3` is clamped to `a`, which is then multiplied by `0.05`
   to yield `0.15` and then added to `b`'s current state -- meaning that the new
   state is `0.15 + 0.15 = 0.3`
and so on and so forth (`b` acts like a non-decaying recurrently additive state).
