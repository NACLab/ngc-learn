# Lesson 2: Building a Model

In this tutorial, we will build a simple model made up of three components: two simple graded cells that are connected 
by a single synaptic cable.

## Instantiating the Dynamical System as a Context

Create a file named `run_lesson2.py` to place/write your Python code below into. 
While building our dynamical system we will set up a `Context` and then add the three different components to it, 
like so: 

```python
from jax import numpy as jnp, random
from ngclearn import Context, MethodProcess
from ngclearn.components import RateCell, HebbianSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist

## create seeding keys
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 4)

## create simple dynamical system: a --> w_ab --> b
with Context("model") as model:
   a = RateCell(name="a", n_units=1, tau_m=0., act_fx="identity", key=subkeys[0])
   b = RateCell(name="b", n_units=1, tau_m=20., act_fx="identity", key=subkeys[1])
   Wab = HebbianSynapse(name="Wab", shape=(1, 1), weight_init=dist.constant(value=1.), key=subkeys[2])
```

Next, we will want to wire together the three components we have embedded into our model, connecting `a` to node `b` 
through synaptic cable `Wab`. In other words, this means that the output compartment of `a` (which, if one checks 
the documentation for `a`, turns out to be `.zF`) must be wired to the input compartment of transformation `Wab`
(i.e., `.inputs`) and the output compartment of `Wab` (i.e., `.outputs`) must be wired to the input compartment 
of `b` (i.e., `.j`). In code, this is done (within the `Context`-block) as follows:

```python                        
    ## wire a to w_ab and wire w_ab to b (a -> Wab -> b)
    a.zF >> Wab.inputs 
    Wab.outputs >> b.j 
```

Finally, to make our dynamical system do something for each step of simulated time, we must append a few basic 
processes (see [Understanding Processes](../configuration/processes.md)) to the context. 
The commands that we will (in general) want will include a `reset` (which will initialize the compartments within 
each node to their "resting" values, i.e., generally zero, if they have them), an `advance` (which moves all the 
nodes one step forward in time according to their compartments' differential equations/internal dynamics), and 
`clamp` (which will allow us to insert data into particular nodes). 
This is simply done by writing the following next (within the `Context`-block):

```python
    ## configure desired commands for simulation object
    reset = (MethodProcess("reset")
             >> a.reset
             >> Wab.reset
             >> b.reset)
    
    advance = (MethodProcess("advance")
               >> a.advance_state
               >> Wab.advance_state
               >> b.advance_state)
    
## set up clamp as a non-compiled utility commands (outside the context-block)
def clamp(x):
    a.j.set(x) ## injects value/tensor x into compartment .j of component a
```

## Running the Dynamical System

With our simple 3-component dynamical system built, we may now apply and run it on a simple sequence of 
one-dimensional real-valued numbers:

```python
## run some data through our simple dynamical system
x_seq = jnp.asarray([[1., 2., 3., 4., 5.]], dtype=jnp.float32)

reset.run()
for ts in range(x_seq.shape[1]):
    x_t = jnp.expand_dims(x_seq[0, ts], axis=0)  ## get data at time ts
    clamp(x_t)
    advance.run(t=ts * 1., dt=1.)
    ## naively extract simple statistics at time ts and print them to I/O
    a_out = a.zF.get()
    b_out = b.zF.get()
    print(" {}: a.zF = {} ~> b.zF = {}".format(ts, a_out, b_out))
```

and, when running your Python script (i.e., `run_lesson2.py`), we should obtain output in your terminal as below: 

```console
$ python run_lesson2.py
 0: a.zF = [1.] ~> b.zF = [[0.05]]
 1: a.zF = [2.] ~> b.zF = [[0.15]]
 2: a.zF = [3.] ~> b.zF = [[0.3]]
 3: a.zF = [4.] ~> b.zF = [[0.5]]
 4: a.zF = [5.] ~> b.zF = [[0.75]]
```

The simple 3-component system simulated above merely transforms the input sequence into another time-evolving series. 
For the curious, in your code above, you modeled a very simple non-leaky integration of cell `b` injected with some 
value produced by `a` (since `Wab = 1`, the synapses had no effect and merely copies the value along). While node 
`a` is always clamped to a value as per the clamp command call we constructed and call above (even though its 
time constant was `tau_m = 0` ms, meaning that it reduces to a stateless "feedforward" cell), `b` had a time constant 
you set to `tau_m = 20` ms. This means, as can be confirmed by inspecting the API for `RateCell`, with your integration time constant `dt = 1` ms:

1. at time step `ts = 0`, the value clamped to `a`, i.e., `1`, was multiplied by    `1/20 = 0.05` and then added 
   `b`'s internal state (which started at the value of `0` through the reset command called before the for-loop);
2. at step `ts = 1`, the value clamped to `a`, i.e., `2`, was multiplied by `0.05` (yielding `0.1`) and then added 
   to `b`'s current state -- meaning that the new state becomes `0.05 + 0.1 = 0.15`;
3. at `ts = 2`, a value `3` is clamped to `a`, which is then multiplied by `0.05` to yield `0.15` and then added to 
   `b`'s current state -- meaning that the new state is `0.15 + 0.15 = 0.3` and so on and so forth (`b` acts like a 
   non-decaying recurrently additive state).
