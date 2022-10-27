# Lesson 2: Custom Cables and Learning Rules

In this (advanced) tutorial, we will focus on the process for writing your own
custom cables. Specifically, you will learn how to sub-class the
[Cable](ngclearn.engine.cables.cable) object and define your own computations
for the key functions that any cable must implement in order to run correctly
within ngc-learn's nodes-and-cables system. Furthermore, you will also learn
how the [Rule](ngclearn.engine.cables.rules.rule) class works as well as when
you will need to sub-class it to write your own Hebbian-like synaptic updates.

We recommend that you create a directory labeled `tutorials/` and a sub-directory
within labeled as `lesson2/` for you to place the code/Python scripts that you will
write in throughout this lesson. Furthermore, it would be useful to read through
the first lesson [Lesson 1](../tutorials/lesson1.md), i.e., notably its first
sub-section "Theory: Cable Theory and Neural Compartments", to be comfortable
with at least the jargon associated with ngc-learn's nodes-and-cables system.
Please note that, in this lesson, we denote the Hadamard product as
$\odot$, $\cdot$ as matrix/vector multiplication, and $(\mathbf{Q})^T$ as
taking the transpose of matrix $\mathbf{Q}$.

## Dissecting the Cable Class

Before writing your own cables, you will first need to understand how the base
cable class works.
We will use the key workhorse cable that ngc-learn uses,
the [DCable](ngclearn.engine.cables.dcable) (the "dense cable"), as our running
example.
A `Cable` in ngc-learn is strictly meant to relay information between
two [Node](ngclearn.engine.nodes.node) objects and can be as simple as
copying/moving information from one source node to a destination node (as in
the "simple cable" `SCable` that ngc-learn implements) or more complex like
a `DCable` which applies transformations on the signals that move across
the synapes it defines. More complex transformations could be designed with
this notion of a cable, for example, one could craft a series of transforms that
attempt to model the more intricate calculations of the dendritic trees that
are known to feed into any single neuron in the brain.

The base `Cable` class requires only a few key arguments to be set, as seen
in its documentation, i.e., `cable_type`, `inp`, `out`, as well as an optional
`name` and `seed`. `cable_type` is simply a string label that uniquely identifies
the cable's "type" within ngc-learn's nodes-and-cables system (and mostly helpful
for guiding its graph plotting visualization modules) while  `inp` and `out`
are needed for defining the two nodes that a `Cable` connects. A cable must, in
line with its conceptual definition, connect two points/nodes and any new type
of cable that one designs is assumed to do this in order to integrate easily
with the nodes-and-cables system.
Looking inside of the [DCable](ngclearn.engine.cables.dcable) source code, we can
see that it indeed respects the base cable's specifications in the snippet below:

```python
# NOTE: inp and out, as per prior tutorials, are tuples of the form
#       (physical node, compartment name string) -- see the Cable documentation for details
def __init__(self, inp, out, init_kernels=None, shared_param_path=None,
             clip_kernel=None, constraint_kernel=None, seed=69, name=None):
    cable_type = "dense" ## label this cable type
    super().__init__(cable_type, inp, out, name, seed)
```

where we see that it has been labeled with the tag "dense" and properly
inserts into the `Cable` parent class the required base arguments. Note that
`Cable` offers the ability to directly name the cables which can be useful
for downstream visualization and also supports an integer `seed` argument
which is useful for controlling global determinism if the target cable is
to include any randomly initialized synapses. Furthermore, notice in the snippet above
that a `DCable` includes additional arguments that are specific to the case of
a "dense" cable transformation, including some convenience arguments that allow
the user to input various "kernel" (or dictionary) arguments that induce
functions such as synaptic value clipping (`clip_kernel`) or synaptic constraints
(`constraint_kernel`). We point out these two particular arguments as it might
be useful for you to include and implement them yourself in order to make use
of these two very common operations used in `NGCGraph` models.
Finally, in general, pointing out these extra cable-specific arguments highlights
the fact that, when writing your own cables,
you can include whatever kernels or base arguments you like but it is important
to define your own block comment/documentation header that carefully and
concretely specifies what each argument or kernel does.

There are several routines that any sub-class `Cable` needs to implement, i.e.,
`compile()` and `propagate()`, as well as a few routines that it <i>could</i> implement,
e.g., `get_params()`, `set_update_rule()`, `calc_update()`, `set_constraint()`,
and `apply_constraints()`, depending on what the cable is intended to do.
Let's break these down briefly before diving into the lesson's example:
1) `compile()` essentially allows ngc-learn to compile the cable as is needed
while `propagate()` implements cable-specific logic for transmitting information
across its internal synapses.
2) `get_params()` is a useful convenience function
for returning any internal parameter matrices that define your cable (it's a
good idea and strongly recommended to implement it).
3) `set_update_rule()` and `calc_update()` are required if you want your cable
to evolve with time (we will cover these later in this lesson), and
4) `set_constraint()` and `apply_constraints()` are very useful convenience
functions to implement if your cable is to evolve and you want ngc-learn to
ensure certain constraints/priors over cables are enforced (we will not cover
these last two in this tutorial, but refer the reader to [DCable](ngclearn.engine.cables.dcable)
for some examples of their implementation).

## Example: Creating a Compound Cable for Modeling Sparse Synaptic Structures

Now that we understand the base specification of a `Cable`, we are now equipped
to design a new one. We will demonstrate this process by crafting a simple,
new kind of cable, which we will call the "sparse" cable (this means we will
choose to set the variable `cable_type = "sparse"`). To implement this cable,
we first consider what we want it to do -- given that synaptic structures in the
brain are sparse, i.e., not every neuron directly wires to every other neuron
(a single neurons receives tens of thousands of inputs within an overall system
of billions of neurons), we will want to ensure that the synaptic matrix connecting
one node (or block of neurons in ngc-learn's lingo) is explicitly forced to
have zero-value synapses. The way in which we will enforce this sparsity during
simulation is to include a binary masking matrix `M` that is constantly applied
to the base matrix `A` (via the equation $\mathbf{V} = \mathbf{A} \odot \mathbf{M}$)
that transforms and transmits the `inp` node's activity to the `out` node's activity.

Let us now design this new cable from scratch step-by-step, implementing in a minimalist  
fashion the basic functions we would want this cable to definitely perform (this
means we will be restricting how general this new cable is, but for the purposes of
writing a clear, easy-to-follow lesson, we eschew how to write a multi-purpose
cable as complex as the `DCable`, which was designed to be flexible for a variety
of neurobiological simulation cases, and assume that the reader will engage in
careful design practice if a truly general cable is desired or needed).

Let us start by creating a file for the code in ngc-learn's `engine` source code folder,
specifically within `ngclearn/engine/cables/` like so:

```console
$ cd ngclearn/engine/cables/
$ touch sparse_cable.py # create the file for our new cable
```

Now, inside of sparse cable file, let us write some basic constructor code at the
top of the file:

```python
import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils
from ngclearn.engine.cables.cable import Cable
from ngclearn.utils.stat_utils import sample_bernoulli

class SparseCable(Cable):
    def __init__(self, inp, out, init_kernels=None, constraint_kernel=None,
                 seed=69, name=None):
        cable_type = "sparse" ## label this cable type
        super().__init__(cable_type, inp, out, name, seed)

        # b/c we sub-class the Cable class, we can retrieve dimensions of input/output nodes
        in_dim = self.src_node.dim # in Cable, "inp" is set to be "self.src_node" (source node)
        out_dim = self.dest_node.dim # in Cable, "out" is set to be "self.dest_node" (destination node)

        self.is_learnable = True ## this will be important later in the tutorial

        ## cable parameters
        self.params["A"] = None # base synaptic matrix A for transforming
        self.params["M"] = None # our binary masking matrix M

        ## do something with your extra, cable-specific constructor arguments
        A_init = init_kernels.get("A_init") # get A's init scheme
        A = transform_utils.init_weights(kernel=A_init, shape=[in_dim, out_dim], seed=self.seed)
        A = tf.Variable(A, name="A_{0}".format(self.name))
        self.params["A"] = A

        p = tf.ones([in_dim, out_dim]) * 0.5
        mask = sample_bernoulli(p) # samples binary matrix of shape p.shape, where >= 0.5 is set to 1
        M = tf.Variable(mask, name="M_{0}".format(self.name))
        self.params["M"] = M
```

As you can see above in our current code, we only expect a sparse cable to contain
a single dense matrix `A` and a binary masking matrix `M`, both set to be of
the exact same size (since we will be applying `M` directly to `A` by elementwise
multiplication).

The first function we need to implement in our `SparseCable` class is the `compile()`
routine, which is typically not too complicated given that this function is only
meant for book-keeping and can use its parent's class's `compile()` to get most
of the work done, like so:

```python
    def compile(self):
        info = super().compile() # use parent class' compile() routine to start
        ## do SparseCable-specific stuff for the rest of this function
        A = self.params.get("A")
        M = self.params.get("M")
        if A is None:
            print("ERROR: synaptic matrix A does not exist!")
        # update the compilation dictionary to include specific info for SparseCable
        info["A.shape"] = A.shape
        info["M.shape"] = M.shape
        return info
```

Note that the `compile()` routine is also your place to insert some correctness
checking code, for example, you could add in some verification code that the
shape of matrix `A` and `M` are shaped correctly given the `self.src_node` and
`self.dest_node` nodes that your cable is meant to transmit information between.

The final, and arguably most important, function that you need to implement is
`propagate()` which will tell our new cable what to do with information from its
source `self.src_node` and how to get it to its destination `self.dest_node`.
We will restrict ourselves to a very simple linear transformation for sake of clarity
(as well as for ease of writing/using a local Hebbian update rule as we will
discuss later in this section), like so:

```python
    def propagate(self):
        ## get your input information from the source node
        # Notice that we obtain the vector of signal values from self.src_node
        # by further calling the node's .extract() function, which takes in
        # as an argument a string indicating what "compartment" of the node
        # to pull information from, i.e., self.src_comp
        in_signal = self.src_node.extract(self.src_comp) # extract input signal
        ## get relevant parameters related to your SparseCable
        A = self.params.get("A")
        M = self.params.get("M")
        ## transmit information along this cable from self.src_node
        V = A * M # enforce the sparsity encoded in M
        out_signal = tf.matmul(in_signal, V) # apply final linear transformation
        return out_signal
```

With your above `propagate()` routine implements, you now have a fully-functioning
cable. However, while technically optional, it's a good idea to implement a few
of the extra routines we mentioned earlier, specifically given that our cable
includes synaptic parameter matrices (`A` and `M`) that we would like the `NGCGraph`
simulation object to be aware of. This is easy to do as shown in the following
snippet:

```python
    def get_params(self, only_learnable=False):
        cable_theta = []
        for pname in self.params:
            if only_learnable == True:
                if self.update_terms.get([pname]) is not None:
                    cable_theta.append(self.params[pname])
            else:
                cable_theta.append(self.params[pname])
        return cable_theta
```

where the above function simply returns a list of the physical matrices
that define our `SparseCable`. Of course, we have included some extra bits of code
in the above routine which will become clearer later when we talk about
synaptic update rules, but for now, the function in returns either all of its
parameters, i.e., `A` and `M`, if `only_learnable=False` and only those that
evolve (such as `A`) if `only_learnable=True`.
Another good optional routine to implement from the list earlier is `apply_constraints()`,
which would tell the `NGCGraph` simulation object what constraints or clipping
is to be applied to your cable after every update (Note: we will cover synaptic
updates later in this tutorial). We leave implementation of `apply_constraints()`
as an exercise to the reader (a good example of one possible implementation
can be found in the standard `DCable` source code within ngc-learn's engine folder).

With the above three routines written into your new cable class, you have
everything you need to test out how this basic transformation will work. We will
do this using an extremely simple two node circuit, i.e., two `SNodes` which
we will name `a` and `b`. Graphically, the 2-node circuit is depicted below
(it is nearly identical to one of the 2-node circuits you would have built
in [Lesson 1](../tutorials/lesson1.md)):

```{eval-rst}
.. table::
   :align: center

   +-------------------------------------------------------+
   | .. image:: ../images/tutorials/lesson2/2n_circuit.png |
   |   :scale: 75%                                         |
   |   :align: center                                      |
   +-------------------------------------------------------+
```

Leave the `ngclearn/engine/cables/` directory
(i.e., `$ cd ../../../`) and
create a new file `test_custom_cable.py`. Next, go ahead and re-compile ngc-learn
so that way it is aware of your new additional cable in `ngclearn/engine/cables/`:

```console
$ python setup.py install
```

which will re-compile/set up ngc-learn. In the `test_custom_cable.py` file, write the following code,
which will make use of our custom cable and create a simulation object for us to play with:

```python
import tensorflow as tf
import numpy as np
from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.cables.sparse_cable import SparseCable ## import your custom cable
from ngclearn.engine.ngc_graph import NGCGraph

a = SNode(name="a", dim=4, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=6, beta=1, leak=0.0, act_fx="identity")

init = {"A_init" : ("unif_scale",1.0)}
a_b = SparseCable(inp=(a, "phi(z)"), out=(b, "dz_td"), init_kernels=init,
                  seed=1234, name="a_to_b")
a_b.hard_wire_nodes(short_name="my_cable") ## Note: we give cable a_b a nickname - "my_cable"

print("Cable {} of type *{}* transmits from:".format(a_b.name, a_b.cable_type))
print("Node {}.{}".format(a_b.src_node.name, a_b.src_comp))
print(" to ")
print("Node {}.{}".format(a_b.dest_node.name, a_b.dest_comp))

## build simulation object/graph
circuit = NGCGraph()
circuit.set_cycle(nodes=[a,b]) # make the graph aware of nodes a and b, in that order
circuit.compile(batch_size=1) # assumes future batch sizes will be size 1, unless re-compiled
```

which should print to your terminal the following (if you run `$ python test_custom_cable.py`):

```console
Cable a_to_b of type *sparse* transmits from:
Node a.phi(z)
 to
Node b.dz_td
```

indicating that the nodes-and-cables system is aware of our new custom `SparseCable`
and that this cable will transform and transmit information extracted from
node `a`'s `phi(z)` compartment to node `b`'s `dz_td` compartment. (Also,
recall from [Lesson 1](../tutorials/lesson1.md) that because we are going
to use the `NGCGraph` static graph internal compilation, our circuit will
expect sample inputs of batch size `1` unless you re-compile).

We can simulate our little circuit by adding the following code to the bottom of
`test_custom_cable.py`:

```python
## these two lines will force our cable matrix A to be a matrix ones
#  (allowing for determinism in this tutorial)
a_b = circuit.cables.get("a_to_b")
a_b.params["A"].assign(a_b.params["A"] * 0 + 1.) # override A with matrix of ones

a_val = tf.ones([1, circuit.getNode("a").dim]) # create sensory data point *a_val*
readouts, _ = circuit.settle(
                clamped_vars=[("a","z",a_val)],
                readout_vars=[("b","phi(z)")],
                K=5
              )
b_val = readouts[0][2]
print(" => Value of b.phi(z) = {}".format(b_val.numpy()))
circuit.clear()
```

If we run the above simulation script (`$ python test_custom_cable.py`), we should
get (due to the determinism created by hacking matrix `A` to be a matrix of ones)
the following in your terminal:

```console
 => Value of b.phi(z) = [[15. 15.  5. 15. 10.  5.]]
```

In order to check that we indeed did apply a binary mask to create a sparse structure,
let us write a new file, `test_dense_cable.py`, and create the same circuit but one
that only uses the standard ngc-learn dense cable (which means that this is the
same circuit but with a cable connecting `a` to `b` that is NOT sparse). Write
the following code in your second file (`test_dense_cable.py`):

```python
from ngclearn.engine.cables.dcable import DCable

a = SNode(name="a", dim=4, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=6, beta=1, leak=0.0, act_fx="identity")

init = {"A_init" : ("constant",1.0)}
dcable_cfg = {"type": "dense", "init_kernels" : init, "seed" : 1234}
a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg,
                name="a_to_b", short_name="dense_cable")
## the following lines do the exact same thing as the above
# a_b = DCable(inp=(a, "phi(z)"), out=(b, "dz_td"), init_kernels=init,
#                   seed=1234, name="a_to_b")
# a_b.hard_wire_nodes(short_name="dense_cable")

print("Cable {} of type *{}* transmits from:".format(a_b.name, a_b.cable_type))
print("Node {}.{}".format(a_b.src_node.name, a_b.src_comp))
print(" to ")
print("Node {}.{}".format(a_b.dest_node.name, a_b.dest_comp))

## build simulation object/graph
circuit = NGCGraph()
circuit.set_cycle(nodes=[a,b]) # make the graph aware of nodes a and b, in that order
circuit.compile(batch_size=1) # assumes future batch sizes will be size 1, unless re-compiled

a_val = tf.ones([1, circuit.getNode("a").dim]) # create sensory data point *a_val*
readouts, _ = circuit.settle(
                clamped_vars=[("a","z",a_val)],
                readout_vars=[("b","phi(z)")],
                K=5
              )
b_val = readouts[0][2]
print(" => Value of b.phi(z) = {}".format(b_val.numpy()))
circuit.clear()
```

and if you run your second script (i.e., `python test_dense_cable.py`), you should
get the following terminal output:

```console
 => Value of b.phi(z) = [[20. 20. 20. 20. 20. 20.]]
```

which indeed confirms that our custom sparse cable is doing what we want (the
values in the vector output of the second case are higher than the first case, when
we ran `test_custom_cable.py`, due to the fact that there would be zeros in matrix `A` of
the sparse cable due to the sparsity enforced by matrix `M`).

You have successfully created a working circuit that uses your custom cable.

## Update Rules and Cables

There is one final, missing detail in our picture above when creating new cables:
their interaction with a learning rule or synaptic update [Rule](ngclearn.engine.cables.rules.rule).
Right now, our new `SparseCable` above is static -- it would not evolve with time
even though we might want it to.
Fortunately, there are two options/cases available to the designer of new cable types:
1) Use one of ngc-learn's currently implemented synaptic update rules, or
2) Write a custom update rule if a standard rule will simply not work.

<b>NOTE:</b> Earlier, you might have noticed that you wrote the following line into your
custom `SparseCable`: `self.is_learnable = True`. While this was meaningless
for the simulation you ran before in this lesson, it becomes important now
as setting this simple boolean variable to `True` already tells ngc-learn
that this cable has the potential to evolve (which is all that the `NGCGraph`
simulation object needs to know).
Make sure that, as a first step, any time you write an experimental/custom cable,
you set this variable to `True` any time you require the cable to evolve
according to some update rule (which will be defined next).

### Case 1: Integrating an Existing Learning Rule

The first of the above two cases is quite simple but strictly assumes that the designer
has ensured that their new cable's internal synaptic parameters can be updated with
locally-available information contained within its `self.src_node` and `self.dest_node`
data members. To use a standard/pre-designed update rule,
let us continue with our two-node circuit from the last section using our newly
designed `SparseCable`. Since our cable simply manipulates
and transmits information
using a sparse matrix, we can readily apply one of ngc-learn's in-built, most-used
rules, i.e, the [HebbRule](ngclearn.engine.cables.rules.hebb_rule). Recall that our sparse
cable also has the nice property that its binary mask `M` merely enforces a constraint on
matrix `A` via multiplication, hence `M` should not evolve. This means,
by our design, we can simply update matrix `A` with a Hebbian rule and be confident
that our static matrix `M` will always ensure that the right numbers in `A` will
be ignored, i.e., set to `0`.  
(NOTE: that if matrix `M` were to be learnable for
some reason, even though it does not make sense for it to evolve in the context of
our sparse cable, then we might need to consider the second case of writing our
own custom update rule.)

First, we need to modify our `sparse_cable.py` in the `ngclearn/engine/cables/`
directory. Edit your file to first write/implement the following:

```python
def set_update_rule(self, preact=None, postact=None, update_rule=None, gamma=1.0,
                    use_mod_factor=False, param=None, decay_kernel=None):
    if update_rule is None:
        print("ERROR: must provide a non-None update rule!")
        sys.exit(1)
    ## integrate update rule into this cable
    if param is not None: # assuming param argument is NOT None
        for pname in param: # for every parameter name in param list
            update_rule.set_terms(terms=[preact, postact], weights=[1.0,-1.0])
            update_rule.point_to_cable(self, pname)
            # Note: .clone() deep copies this rule for pname
            self.update_terms[pname] = update_rule.clone()
```

and second, add the following code at the bottom:

```python
def calc_update(self):
    delta = []
    if self.is_learnable == True:
        A_update_rule = self.update_terms.get("A")
        dA = A_update_rule.calc_update()
        delta.append(dA)
    return delta
```

Notice that the above two routines were earlier described as "optional" but now
<i>must</i> be implemented if the a custom cable is to be
learnable. Notice that we have provided the barest minimum possible implementation
of a learning rule for our `SparseCable` with only a small error check in
`set_update_rule()` (see `DCable` for more intricate checks).
In general, it is up to the designer on how to craft the synaptic update rules, but
<i>some key design requirements include</i>:
1) You must implement the setter routine `set_update_rule()` which might or might
not use the convenience arguments (e.g., `preact`, `postact`, `param`, etc.) that
this routine implements and you should utilize the `Cable` parent class's
`self.update_terms` dictionary as it allows you to insert parameter names, such
as `A` in our example, and the learning rule they are to map to/use (in our case
`update_rule`, provided as an argument to `set_update_rule()`), and
2) You must implement the update routine `calc_update()`, which takes
whatever global and local variables you configured in `set_update_rule()`, such as
the learning rules stored in the dictionary `self.update_terms` and applies them
to create the update matrices that will be returned in a list data structure `delta`
(note that this function MUST return a list of updates of the same shape as their
corresponding parameter matrices, as this list will often be used in a Tensorflow
optimizer to physically change the synapses externally).

As usual, after you write the above, make sure you leave the ngclearn core directory
and re-compile ngc-learn by running `$ python setup.py install`.
We can ensure that our custom sparse cable `a_b` evolves with time by writing a new
file, `test_custom_learned_cable.py`. First, add in the following initialization code:

```python
import tensorflow as tf
import numpy as np
from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.cables.sparse_cable import SparseCable ## import your custom cable
from ngclearn.engine.ngc_graph import NGCGraph
from ngclearn.engine.cables.rules.hebb_rule import HebbRule

a = SNode(name="a", dim=4, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=6, beta=1, leak=0.0, act_fx="identity")

init = {"A_init" : ("constant",0.05)}
a_b = SparseCable(inp=(a, "phi(z)"), out=(b, "dz_td"), init_kernels=init,
                  seed=1234, name="a_to_b")
a_b.hard_wire_nodes(short_name="my_cable") ## Note: we give cable a_b a nickname - "my_cable"

## set up a simple (two-factor) Hebbian update
# the two lines below tell ngc-learn that we only want to update matrix A
# using pre-synaptic values phi(z) of node a and post-synaptic values phi(z) of node b
rule = HebbRule()
a_b.set_update_rule(preact=(a,"phi(z)"), postact=(b,"phi(z)"),
                    update_rule=rule, param=["A"])

## build simulation object/graph
circuit = NGCGraph()
circuit.set_cycle(nodes=[a,b]) # make the graph aware of nodes a and b, in that order
circuit.compile(batch_size=1) # assumes future batch sizes will be size 1, unless re-compiled
```

Then, to run/simulate our circuit and evolve the synapses, we can then write the
following bit of code at the bottom of the `test_custom_learned_cable.py`:

```python
## run simulation with evolving sparse cable
n_iter = 4
opt = tf.keras.optimizers.SGD(0.05)
## adjust 2-node circuit with Hebbian learning
for t in range(n_iter):
    a_val = tf.ones([1, circuit.getNode("a").dim])
    readouts, delta = circuit.settle(
                        clamped_vars=[("a","z",a_val)],
                        readout_vars=[("b","phi(z)")],
                        K=5, calc_delta=True
                      )
    b_val = readouts[0][2]
    print(" => Dense Value of b.phi(z) = {}".format(b_val.numpy()))
    opt.apply_gradients(zip(delta, circuit.theta))
    circuit.clear()
print()
```

where we see that we have import ngc-learn's in-built [HebbRule](ngclearn.engine.cables.rules.hebb_rule)
which simply computes an update to a matrix through a (potentially-weighed)
two-factor rule, i.e., in our case, it will end up being:
$\Delta A = -\mathbf{z}_{pre} \cdot (\mathbf{z}_{post})^T$ where
$\mathbf{z}_{pre}$ is the chosen pre-cable/synaptic compartment value (`phi(z)`) and
$\mathbf{z}_{post}$ is the chosen post-cable/synaptic compartment value (`phi(z)`).

If you run your script `test_custom_learned_cable.py`, you should get the
following terminal output:

```console
=> Dense Value of b.phi(z) = [[0.75 0.75 0.25 0.75 0.5  0.25]]
=> Dense Value of b.phi(z) = [[1.3125 1.3125 0.3125 1.3125 0.75   0.3125]]
=> Dense Value of b.phi(z) = [[2.296875 2.296875 0.390625 2.296875 1.125    0.390625]]
=> Dense Value of b.phi(z) = [[4.0195312  4.0195312  0.48828125 4.0195312  1.6875     0.48828125]]
```

Note that, for mere demonstration purposes, the above code simply adjusts
the values of our cable `a_b` by Hebbian learning, meaning that the weight
values in matrix `A` will simply increase the correlational strength
between node `a` and `b`
(i.e., it is not optimizing an error/loss as we have done in other examples).
Note that the above rule is rather silly, as all it will do is keep increasing
the synaptic weight values each time `.settle()` is called. It is assumed that
the problem context will help the user to determine the proper circuit structure
and what learning rule/form is needed.

### Case 2: Writing a Custom Learning Rule

The second case is a bit more involved and difficult than the first case. Specifically,
if you design a cable with internal parameters that cannot be simply adjusted
by local signals (at least directly as in our example above, for example, you
have two linear transforms applied in a chain on values extracted from node
`a`), then you will need to implement your own synaptic update rule so that
you can properly take into account the internal structure of your cable. To do
so, you will need to sub-class the [Rule](ngclearn.engine.cables.rules.rule) class,
and, much as you did with the base [Cable](ngclearn.engine.cables.cable) when
crafting a custom `SparseCable`, write your own internal code that will
calculate the needed matrix/vector updates your cable would expect.
While we will not demonstrate how to craft a complicated learning rule (as
such a case is dependent on the user/designer's modeling problem and context),
we will show how to develop a simple alternative custom learning rule
for our `SparseCable` that we have developed throughout this lesson.

Concretely, we will call our new learning rule a `MaskedHebbRule` and simply
design it to utilize the binary mask `M` that should come within the cable it
modifies (b/c our sparse cable does have such a matrix) -- in other words, we will
be designing a very specialized update
rule that would only work with a `SparseCable` or a cable that has internally
implemented some masking matrix `M` (which would be sufficient for our
hypothetical use-case as developed in this lesson). To implement a custom
learning rule, you will need to change into `ngclearn/engine/cables/rules/` and
create a new file `maskhebb_rule.py`. In this file, write the following
header:

```python
import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils
from ngclearn.engine.cables.rules.rule import UpdateRule

class MaskedHebbRule(UpdateRule):
    def __init__(self, name=None):
        rule_type = "masked_hebbian"
        super().__init__(rule_type, name)
```

and then write the two minimum-required routines:

```python
def calc_update(self, for_bias=False):
    ## extract statistics needed for this rule
    preact_node = self.cable.src_node
    preact_term = preact_node.extract("phi(z)")

    postact_node = self.cable.dest_node
    postact_term = postact_node.extract("phi(z)")

    M = self.cable.params["M"]

    ## calculate the final synaptic update matrix
    update = -tf.matmul(preact_term , postact_term, transpose_a=True) * M
    return update
```

and finally implement/write at the bottom:

```python
def clone(self):
    rule = MaskedHebbRule(self.name)
    rule.cable = self.cable
    rule.param_name = self.param_name
    return rule
```

with the above, we implemented the most basic form of a cable learning rule.
The `.clone()` method is merely a duplication that ngc-learn requires/uses in order
to allow the user to re-use the rule across different cables (without causing
pointer collisions). The more important routine is `.calc_update()` which is
where the designer will largely want to write the equations/dynamics that they
desire in order to compute an adjustment matrix/vector for the target cable.
Note that in our very simple rule above, all we did was calculate a simple
Hebbian update (multiplying the `phi(z)` compartment of the pre-synaptic/cable
node with the `phi(z)` compartment of the post-synaptic/cable) and then
multiply by the very same binary matrix `M` that is contained within the sparse
cable that this rule would modify.

Note that our custom rule above is very brittle
in that it would only work with a cable that contains the matrix `M` within it
(other cables such as `DCable` or `SCable` do not) but often all we require
for a research project is a very specific update for a very specific cable that
we are experimenting with.
To implement more general update rules, it is recommended that the user/designer
studies the source code of ngc-learn's cables such as [DCable](ngclearn.engine.cables.dcable)
and [SCable](ngclearn.engine.cables.scable), its varous nodes
(e.g., `SNode`, `FNode`, etc., given that different nodes might have different compartments),
as well as its standard learning rules [HebbRule](ngclearn.engine.cables.rules.hebb_rule).

To try out our new learning rule, move out of the current directory again
(i.e., `cd ../../../../`) and
re-compile ngc-learn so it is aware of your custom-designed rule (`$ python setup.py install`).
Then create a file called `test_custom_rule.py` and write the following:

```python
import tensorflow as tf
import numpy as np
from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.cables.sparse_cable import SparseCable ## import your custom cable
from ngclearn.engine.ngc_graph import NGCGraph
from ngclearn.engine.cables.rules.maskhebb_rule import MaskedHebbRule

a = SNode(name="a", dim=4, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=6, beta=1, leak=0.0, act_fx="identity")

init = {"A_init" : ("constant",0.05)}
a_b = SparseCable(inp=(a, "phi(z)"), out=(b, "dz_td"), init_kernels=init,
                  seed=1234, name="a_to_b")
a_b.hard_wire_nodes(short_name="my_cable") ## Note: we give cable a_b a nickname - "my_cable"

## set up a simple (two-factor) Hebbian update
# the two lines below tell ngc-learn that we only want to update matrix A
# using pre-synaptic values phi(z) of node a and post-synaptic values phi(z) of node b
rule = MaskedHebbRule()
a_b.set_update_rule(update_rule=rule, param=["A"])

## build simulation object/graph
circuit = NGCGraph()
circuit.set_cycle(nodes=[a,b]) # make the graph aware of nodes a and b, in that order
circuit.compile(batch_size=1) # assumes future batch sizes will be size 1, unless re-compiled

print(" >> Simulating custom learning rule on custom cable!")
## run simulation with evolving sparse cable
n_iter = 4
opt = tf.keras.optimizers.SGD(0.05)
## adjust 2-node circuit with Hebbian learning
for t in range(n_iter):
    a_val = tf.ones([1, circuit.getNode("a").dim])
    readouts, delta = circuit.settle(
                        clamped_vars=[("a","z",a_val)],
                        readout_vars=[("b","phi(z)")],
                        K=5, calc_delta=True
                      )
    b_val = readouts[0][2]
    print(" => Dense Value of b.phi(z) = {}".format(b_val.numpy()))
    opt.apply_gradients(zip(delta, circuit.theta))
    circuit.clear()
print()
```

Notice the code above is very similar to `test_custom_learned_cable.py` but
modified to use your new local update rule instead.
If you run this script, you should get the exact same output as
you did for the last section/earlier:

```console
>> Simulating custom learning rule on custom cable!
=> Dense Value of b.phi(z) = [[0.75 0.75 0.25 0.75 0.5  0.25]]
=> Dense Value of b.phi(z) = [[1.3125 1.3125 0.3125 1.3125 0.75   0.3125]]
=> Dense Value of b.phi(z) = [[2.296875 2.296875 0.390625 2.296875 1.125    0.390625]]
=> Dense Value of b.phi(z) = [[4.0195312  4.0195312  0.48828125 4.0195312  1.6875     0.48828125]]
```

Notice that since our naive custom learning rule simply computed a
Hebbian product and multiplied this by the masking matrix `M`, when running our
code we obtained the same result as we did when we used the standard in-built
Hebbian learning rule in the last section (Case 1). This is due to the fact that
our custom update
$\Delta \mathbf{A} = -( \mathbf{z}_{pre} \cdot (\mathbf{z}_{post})^T ) \odot \mathbf{M}$
applied to the matrix $\mathbf{A}$, i.e., $\mathbf{A} \leftarrow \mathbf{A} + 0.05 \Delta \mathbf{A}$,
is the same as the original update because we always multiply $\mathbf{A}$ by $\mathbf{M}$
within our cable, i.e., $\mathbf{V} = \mathbf{A} \odot \mathbf{M}$, before
propagating information across the cable's synapses (meaning that matrix $\mathbf{M}$
restricts matrix $\mathbf{V}$ to have zeros in the exact same spot).
In other words:

$$
\mathbf{A} &\leftarrow \mathbf{A} + 0.05 \big( -\mathbf{z}_{pre} \cdot (\mathbf{z}_{post})^T \big) \\
\mathbf{V} &= \mathbf{A} \odot \mathbf{M}
$$

results in the same computation as:

$$
\mathbf{A} &\leftarrow \mathbf{A} + 0.05 \big( -\mathbf{z}_{pre} \cdot (\mathbf{z}_{post})^T \big) \odot \mathbf{M} \\
\mathbf{V} &= \mathbf{A} \odot \mathbf{M}
$$

the only difference is that our custom rule (second set of equations above) just enforces the sparsity
constraint directly on the update.
Note: You might also notice that in the second set
of equations above that, given our custom learning rule, we could also remove
the redundant multiplication of `M` (making the 2nd equation simply be
$\mathbf{V} = \mathbf{A}$) by swapping out our custom sparse cable with
ngc-learn's standard dense cable and ensuring that its matrix `A` is initialized
according to a scheme that involves the binary matrix `M`, e.g.,
$\mathbf{A}_{ij} = \big( \sim \mathcal{N}(\mu,\sigma)\big) \mathbf{M}_{ij}$.

You have now implemented your own custom learning rule for your custom cable. Armed
with this knowledge, you are now ready to go forward and design your own
cables if what is provided by ngc-learn's in-built tools is not enough!
<i>A word of caution</i>:
Implementing one's own cables and learning rules is
an advanced topic and this tutorial only covered the very basics of doing so.
To properly design new cables and rules, it is important that the designer
follow at least two steps:
1) Understand the basic calls required by ngc-learn globally (we discussed a few
key ones, but there are some others you can find for certain cases in
[Cable](ngclearn.engine.cables.cable) and
[Rule](ngclearn.engine.cables.rules.rule)),
2) Work out the mathematics behind what needs to be done on paper a priori,
specifically ensuring that they have developed the linear algebraic form of
their update equates in order to take advantage of the efficient GPU simulation core
that drives the `NGCGraph` simulation object.
For example, in the `.calc_update()` that you implemented above, the coding logic that you
could write inside is quite open-ended and, as we saw in our example, does
require knowledge of what is going on in the cable that we would like our
rule to modify (as well as the nodes that it connects).


## Final Thoughts on Custom Cables and Learning Rules

While this lesson went through the basic process/pipeline of designing a custom
cable and its (learning) update, crafting one's own cables and rules is not
a simple task and generally depends on the user/designer to understand
their problem context, their neural circuit(s), and the learning dynamics
that they wish to model. There are many possible ways to engineer learning rules
within the context of the design framework presented above and despite the progress
that has made over the decades, biologically-motivated
learning is still in its infancy (with respect to both statistical learning
and computational neuroscience), despite the plethora of ideas that have been
proposed and studied.

ngc-learn is meant to provide the modeler a platform to implement and simulate
their ideas with respect to stateful neural circuits (such as those that
characterize predictive coding or spiking neural systems). However, there is
no guarantee that any new rule or cable structure that is implemented will be stable
and work well in the context of any given problem. It is important that the user
conducts many experimental trials/tests and makes comparisons to known
stable rules (such as error-driven Hebbian learning) in order to determine the validity
of their idea(s) -- when designing learning rules, it is important to "fail fast"
so you can fruitfully iterate over your idea/formulation.

<i><b>A Parting Remark</b></i>: while this lesson has demonstrated how the user can
implement their own cable and its update rule in ngc-learn, if the user desires the
cable and/or learning rule that they created (using ngc-learn) to be officially implemented
into the library itself (which we enthusiastically welcome!), please see the [contributing guidelines](https://github.com/ago109/ngc-learn/blob/main/CONTRIBUTING.md) and
reach out to the Neural Adaptive Computing (NAC) Laboratory (contact:
`ago@cs.rit.edu`) to discuss its integration.
