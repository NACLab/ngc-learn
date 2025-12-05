# Lesson 3: Evolving Synaptic Efficacies

In this tutorial, we will extend a model context/controller with three components, two cell components connected with a 
synaptic cable component, to incorporate a basic a two-factor Hebbian adjustment process.

## Adding a Learnable Synapse to a Multi-Component System

Create a Python script/file named `run_lesson3.py` to place/write your Python code below into. 
Let us start by building a controller/model-context similar to previous lessons with the one exception that now we will 
trigger the synaptic connection between `a` and `b` to adapt via a simple 2-factor Hebbian rule. This Hebbian rule will 
require us to wire the output compartment of `a` to the pre-synaptic compartment of the synapse `Wab` and the output 
compartment of `b` to the post-synaptic compartment of `Wab`. This will wire in the two relevant factors needed to 
compute a simple Hebbian adjustment.

We do this specifically as follows:

```python
from jax import numpy as jnp, random, jit
from ngclearn import Context, MethodProcess
from ngclearn.components import HebbianSynapse, RateCell
from ngclearn.utils.distribution_generator import DistributionGenerator as dist

## create seeding keys
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

## create simple system with only one F-N cell
with Context("Circuit") as circuit:
    a = RateCell(name="a", n_units=1, tau_m=0., act_fx="identity", key=subkeys[0])
    b = RateCell(name="b", n_units=1, tau_m=0., act_fx="identity", key=subkeys[1])

    Wab = HebbianSynapse(
        name="Wab", shape=(1, 1), eta=1., sign_value=-1., weight_init=dist.constant(value=1.), 
        w_bound=0., key=subkeys[3]
    )

    # wire output compartment (rate-coded output zF) of RateCell `a` to input compartment of HebbianSynapse `Wab`
    a.zF >> Wab.inputs 
    # wire output compartment of HebbianSynapse `Wab` to input compartment (electrical current j) RateCell `b`
    Wab.outputs >> b.j 
    
    # wire output compartment (rate-coded output zF) of RateCell `a` to presynaptic compartment of HebbianSynapse `Wab`
    a.zF >> Wab.pre 
    # wire output compartment (rate-coded output zF) of RateCell `b` to postsynaptic compartment of HebbianSynapse `Wab`
    b.zF >> Wab.post 
    
    ## create and compile core simulation commands  
    evolve = (MethodProcess("evolve")
              >> a.evolve)
    
    advance = (MethodProcess("advance")
               >> a.advance_state)
    
    reset = (MethodProcess("reset")
             >> a.reset)
    
## set up non-compiled utility commands
def clamp(x):
    a.j.set(x)
```

Now with our simple system above created, we will now run a simple sequence of one-dimensional "spike" data through it 
and evolve the synapse every time step like so:

```python
## run some data through the dynamical system
x_seq = jnp.asarray([[1, 1, 0, 0, 1]], dtype=jnp.float32)

reset.run()
print("{}: Wab = {}".format(-1, Wab.weights.value))
for ts in range(x_seq.shape[1]):
  x_t = jnp.expand_dims(x_seq[0,ts], axis=0) ## get data at time t
  clamp(x_t)
  advance.run(t=ts*1., dt=1.)
  evolve.run(t=ts*1., dt=1.)
  print(" {}: input = {} ~> Wab = {}".format(ts, x_t, Wab.weights.get()))
```

After running `run_lesson3.py`, your code should produce (printed to I/O) the same output as below:

```console
-1: Wab = [[1.]]
 0: input = [1.] ~> Wab = [[2.]]
 1: input = [1.] ~> Wab = [[4.]]
 2: input = [0.] ~> Wab = [[4.]]
 3: input = [0.] ~> Wab = [[4.]]
 4: input = [1.] ~> Wab = [[8.]]
```

Notice that for every non-spike (a value of `0`), the synaptic value remains the same (because the product of a 
pre-synaptic value of `0` with a post-synaptic value of anything -- in this case, also a `0` -- is simply `0`, meaning 
that no change will be applied to the synapse). For every spike (a value of `1`), we get a synaptic change equal to 
`dW = input * (Wab * input)`; so for the first time-step, the weight will change according to 
`W = W + eta * dW = W + dW` and `dW = 1 * (1 * 1) = 1`, whereas, for the second time-step, `W` will be increased by 
`dW = 1 * (2 * 1) = 2` (yielding a new synaptic strength of `W = 4`).

As per the above, you have now created your first plastic, evolving neuronal system!
