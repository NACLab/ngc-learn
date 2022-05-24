# Lesson \# 1: The Nodes-and-Cables System

In this tutorial, we will focus on working through the very basics of ngc-learn's
nodes-and-cables system. Specifically, you will learn how various (mini-)circuits
are built in order to develop an intuition of how these fundamental
modeling blocks fit together and how, when they are put together in the right way,
you can simulate your own evolving dynamical neural systems.

We recommend that you create a directory labeled `tutorials/` and a sub-directory
within labeled as `lesson1/` for you to place the code/Python scripts that you will
write in throughout this lesson.

## Theory: Cable Theory and Neural Compartments

At its core, part of ngc-learn's core design is inspired by (neural)
<a href="http://www.scholarpedia.org/article/Neuronal_cable_theory">cable theory </a>,
where neuronal units, which are arranged in complex connectivity structures,
are viewed as performing dendritic calculations (of varying complexity).
In essence, a particular neuron integrates information from different input signal
sources (for example, signals produced by other neurons), in often highly nonlinear
ways through a complex dendritic tree.

Although modeling a complete neuronal system through the lens of cable theory is
complex and intricate in of itself, ngc-learn is built with this direction in mind.
ngc-learn starts with with the idea a neuron (or a cluster of them) can be viewed
as a node, or [Node](ngclearn.engine.nodes.node) (also see {ref}`node-model`),
and each bundle of synapses that connect pairs of nodes can be viewed as a cable,
or [Cable](ngclearn.engine.cables.cable) (also see {ref}`cable-model`).

Each node has multiple, different (named) "compartments", which are regions
or slots within the node that other nodes can deposit information/signals into.
These compartments allow a node to collect information from many different
connected/related nodes and then decide how to combine these different signals
in order calculate its own output activity (either in the form of a rate-coded
firing rate or binary spikes) using the integration logic defined within its
own specific `step()` function. When an NGC system, composed of many of these
nodes, is simulated over a period of time (processing some form of sensory input),
its underlying simulation object (the `NGCGraph`) calls the `step()` routine
of each constituent node within one discrete time step. The order in which the
node `step()` routines are called is governed by "execution cycles", which are
defined by the experimenter at object initialization, for example, a user
might want all of the state nodes to first execute their internal step logic
before the error nodes do (which can be done by specifying two distinct cycles
in the order desired).

As a result, many nodes and cables result in an NGC system where each
node is itself, in general, a stateful computation even if we are processing
inherently non-temporal data such as static images.

## Node and Cable Fundamentals

To start creating predictive processing models and other neurobiological neural
systems, we must first examine the fundamental building blocks you will need to
craft them.
At a high level, motivated by the theory described above, an NGC system is made
up of multiple nodes and cables where each node (or cluster/group of neurons)
in the system contains one or more compartments (or vectors of scalar numbers/signals)
and each cable transmits the information (vector of numbers) inside one compartment
within one node and transforms this information (potentially with synapses) and
finally deposits this transformed information into one single compartment of another node.
Understanding how nodes and cables relate to each other in ngc-learn is necessary
if one wants to build and simulate their own custom NGC system (for example, the
arbitrary 3-node one graphically depicted in the figure below).

```{eval-rst}
.. table::
   :align: center

   +-------------------------------------------------------------+
   | .. image:: ../images/tutorials/lesson1/nodes_and_cables.png |
   |   :scale: 42%                                               |
   |   :align: center                                            |
   +-------------------------------------------------------------+
```

### The Node

First, let us examine the [node object](ngclearn.engine.nodes.node) itself.
A node (or `Node`) contains inside of it a cluster (or block) of neurons,
the number of which is controlled through the `dim` argument. We will, in this
tutorial lesson, examine two core node types within ngc-learn, the stateful
node (or [SNode](ngclearn.engine.nodes.snode)) and the error node
([ENode](ngclearn.engine.nodes.enode)), although there are other node types
(such as convenience nodes like the `FNode` or spiking nodes).

Every node in ngc-learn has several compartments, which are made explicit
in each node's documentation listed under "Compartments:" and the names of
which can be programmatically accessed through the node's data member
`.compartment_names`. As mentioned in the last section, the signal values within
these compartments are often combined together according to the logic defined
within a node's `.step()` simulation function. Furthermore, each node contains
two other data members of particular interest -- the `.connected_cables` list
and the `.constant_names` list. The `.constant_names` contains fixed integer/scalar
coefficients/values that are also used within an node's `.step()` logic, such
as biological constants derived from experimental data or user-set coefficients
that can be defined before simulation (like an integration time constant).
The `.connected_cables` is an (unordered) list of `Cable` objects that connect
to a particular node (one can iterate over this list and print out the names of
each cable if need be).
Each cable object, which we will discuss in more detail
later, has knowledge of the specific compartment within a given node it is to
deposit its information into and the node can easily query the name of this
compartment by accessing the cable's data member `.dest_comp`.

Given the information above (with the aid of a few other internal book-keeping
data structures), a node, after its own `.compile()` routine has been executed
(which is done within an `NGCGraph`'s `.compile()` function call), will run
its own internal logic each time its `.step()` is called, continually integrating
information from its named compartments until the end of simulation time window.
While we will defer the exact details of how a `.step()` function is/should be
implemented for a subsequent tutorial lesson (which will aid developers
interested in contributing their own node types to ngc-learn), we can briefly
speak to the neural dynamics that occurs within `.step()` for the two nodes you
will work with in this lesson.

For a state node (`SNode`), as seen in its [API](ngclearn.engine.nodes.snode),
we see that we have six compartments, which can be printed to I/O as in the
following code snippet/example (you can place it in a script named `test_node.py`):

```python
import tensorflow as tf
import numpy as np
from ngclearn.engine.nodes.snode import SNode

a = SNode(name="a", dim=1, beta=1, leak=0.0, act_fx="identity")
print("Compartments:  {}".format(a.compartment_names))
```

which will print the compartments internal to the above node `a`:

```console
Compartments:  ['dz_bu', 'dz_td', 'z', 'phi(z)', 'S(z)']
```

<!--i.e., `dz_td`, `dz_bu`, `z`, `phi(z)`, `S(z)`, and `mask`.-->
We will discuss the first four since the last one is a
specialized compartments only used in certain situations. The neural dynamics
of a state node, according to the first four compartments, is mathematically
depicted by the following partial differential equation:

<!--
```
d.z/d.t = -z * leak + dz + prior(z), where dz = dz_td + dz_bu * phi'(z)
```
-->
$$
\frac{\partial \mathbf{z}}{\partial t} = -\gamma_{leak} \mathbf{z} +
(\mathbf{dz}_{td} + \mathbf{dz}_{bu} \odot \phi^\prime(\mathbf{z})) + \mbox{prior}(\mathbf{z})
$$

where we also formally represent the compartments `dz_bu`, `dz_td`, `z`, and `phi(z)`
as $\mathbf{dz}_{bu}$, $\mathbf{dz}_{td}$, $\mathbf{z}$, and $\phi(\mathbf{z})$,
respectively. This means that, if we use Euler integration to update the `SNode`'s compartment
$\mathbf{z}$ (the default in ngc-learn), $\mathbf{z}$ is updated each call to `.step()` as follows:

<!--
```
z <- z * zeta + d.z/d.t * beta, and
phi(z) = f(z), where f is activation function such as tanh(z)
```
-->
$$
\mathbf{z} &\leftarrow \zeta \mathbf{z} + \beta \frac{\partial \mathbf{z}}{\partial t} \\
\phi(\mathbf{z}) &= tanh(\mathbf{z}) \quad \mbox{// $\phi(\mathbf{z})$ can be any activation function}
$$

and finally, after $\mathbf{z}$ is updated, the state node will apply an element-wise
nonlinear function to $\mathbf{z}$ to get $\phi(\mathbf{z})$ (which is also the name of the
fourth compartment). Note that, in the above, we see several of the node's
key constants defined, i.e. $\beta$ or `.beta` (the strength of
perturbation applied to the node's $\mathbf{z}$ compartment), $\gamma_{leak}$ or
`.leak` (the strength of the amount of decay applied to the $\mathbf{z}$ compartment's value),
and $\zeta$ or `.zeta` (the amount of recurrent carry-over or how "stateful" the node is --
if one sets the constant `.zeta = 0`, the node becomes "stateless").
$\mbox{prior}(\mathbf{z})$ just refers to a distribution function that can be applied to
the $\mathbf{z}$ compartment (see [Walkthrough \#4](../walkthroughs/demo4_sparse_coding.md)
for how this is used/set). We see by observing the above differential equation that a
state node is primarily defined by the value of its $\mathbf{z}$ compartment and how
this compartment evolves over time is dictated by several factors including the
other two compartments $\mathbf{dz}_{td}$ and $\mathbf{dz}_{bu}$ ($\phi^\prime(\mathbf{z})$
refers to the first derivative of the `SNode`'s activation function $\phi(\mathbf{z})$
which can be turned off if desired). Note that multiple cables can feed into
$\mathbf{dz}_{td}$ and $\mathbf{dz}_{bu}$ (multiple deposits would be summed to
create a final value for either compartment).

As we can see in the above dynamics equations, a state node is simply a set of
rate-coded neurons that update their activity values according to a linear
combination of several "pressures", notably the two key pressures $\mathbf{dz}_{td}$
(`dz_td`) and $\mathbf{dz}_{bu}$ (`dz_bu`) which are practically identical except
that `dz_bu` is a pressure (optionally) weighted by the state node's activation
function derivative $\phi^\prime(\mathbf{z})$.
In a state node, when you wire other nodes to it, the `.step()` function will
specifically assume that signals are only ever being deposited into either `dz_td` or
`dz_bu` and NOT into $\mathbf{z}$ (or `z`) and $\phi(\mathbf{z})$ (or `phi(z)`), since
these last two compartments being evolved according to the equations presented earlier --
note that if you accidentally "wire" another node to the `z` or `phi(z)` compartments,
the `SNode` will simply ignore those since its `.step()` function only assumes
`dz_td` and `dz_bu` receive signals externally).

With the `SNode` above, you can already build a fully functional NGC system (for
example, a Harmonium as in [Walkthrough \#6](../walkthroughs/demo6_boltzmann.md)),
however, there is one special node that we should also describe that will allow
you to more easily construct arbitrary predictive coding systems. This node is
known as the error node (`ENode`) and, as seen in its [API](ngclearn.engine.nodes.enode),
it contains the following key compartments -- `pred_mu`, `pred_targ`, `z`, `phi(z)`,
and `L` or, formally, $\mathbf{z}_\mu$, $\mathbf{z}_{targ}$, $\mathbf{z}$,
$\phi(\mathbf{z})$, and $L(\mathbf{z})$.
An error node is, in some sense, a convenience node because it is actually
mathematically a simplification of a state node that is evolved over a period
of time (it is a derived "fixed-point" of a pool of neurons that compute
mismatch signals evolved over several simulation time steps) and is
particularly useful when we want to simulate predictive coding systems faster (and
when one is not concerned with the exact biological implementation of neurons that
compute mismatch signals but only with their emergent behavior).

The error node dynamics are considerably simpler than that of a
state node (and, since they are driven by a derived fixed-point calculation,
they are stateless) and simply dictated by the following:

<!--
```
z = pred_targ - pred_mu, and
phi(z) = f(z), where f is activation function such as identity(z)
```
-->
$$
\mathbf{z} &= \mathbf{z}_\mu - \mathbf{z}_{targ} \\
\phi(\mathbf{z}) &= identity(\mathbf{z}) \quad \mbox{// $\phi(\mathbf{z})$ can be any activation function} \\
L(\mathbf{z}) &= \sum_j (\mathbf{z} \odot \mathbf{z})^2_j \mbox{// Gaussian error neurons}
$$

where $\odot$ denotes elementwise multiplication and $\mathbf{z}_{targ}$ (or
`pred_targ`) is the target signal (which can be accumulated from multiple
sources, i.e., if more than cable feeds into it, the set of deposits are summed
to create the final compartment value of `pred_targ`) and $\mathbf{z}_\mu$ or (`pred_mu`) is the
expectation of the target signal (which can also be the sum of multiple deposits
from multiple cables/sources, i.e., multiple deposits from multiple cables
will be summed to calculate the final value of `pred_mu`). Note that for $L(\mathbf{z})$
(or `L`), we only depict one possible form that this compartment can take -- the
Gaussian error neuron (which results in a local mean squared error loss) --
although are forms are possible (such as the Laplacian error neuron).

Below, we graphically depict the `SNode` (Left) and the `ENode` (Right):

```{eval-rst}
.. table::
   :align: center

   +-----------------------------------------------------------+-----------------------------------------------------------+
   | .. image:: ../images/tutorials/lesson1/ngclearn_snode.png | .. image:: ../images/tutorials/lesson1/ngclearn_enode.png |
   |   :scale: 40%                                             |   :scale: 40%                                             |
   |   :align: center                                          |   :align: center                                          |
   +-----------------------------------------------------------+-----------------------------------------------------------+
```

notice that both diagrams indicate that multiple incoming signals (each indicated
by a curved diamond-head arrow) are summed within the cell body compartment they
are deposited into with the $\Sigma$ symbol. In the `SNode`, the signals
`dz_td` and `dz_bu` are combined by addition, i.e., $+$ (in the light blue box), whereas
in the `ENode`, the signals `pred_targ` and `pred_mu` are combined by subtraction,
i.e., $-$ (in the red box) (they are contrasted to produce a mismatch/difference signal).

While we do not touch on it in this tutorial lesson, a user could write their
own custom nodes as well, making sure to subclass the `Node` class and then
define the dendritic calculation that they require within `.step()` and ensuring
that their custom node writes to the `Node` class's core compartment data
structures so that ngc-learn can effectively simulate the node's evolution over
time. Writing one's own custom node will be the subject of an upcoming ngc-learn
tutorial lesson.

### The Cable

Given the above understanding of a node, all that remains is to combine pairs of
them together with an object known as the [cable](ngclearn.engine.cables.cable).
Note that all cables fundamentally are responsible for one particular job:
taking the information in one compartment of one "source node", doing something
to this information (such as transforming with a bundle of synapses via linear
algebra operations), and then depositing this information into the compartment
of another "destination node".
To do this, there are two primary types of cables you should be familiar with:  
the simple cable [SCable](ngclearn.engine.cables.scable) and the dense cable
[DCable](ngclearn.engine.cables.dcable).
The simple cable simply transmits information directly from one node's compartment
to another node's compartment, simply multiplying the information from the source
node by its scalar data member `.coeff` (by default this is set to the value of `1`).
The dense cable, in contrast, is a bit more involved as it takes the information
in one node's compartment and applies some variant of a linear transformation
to this signal before depositing it into the compartment of another node (if you
wanted a cable to do something more complex than this, you could, as you can for
the `Node` class, write your own custom cable, but we leave this as the subject
for a future upcoming tutorial lesson).

Building cables is primarily done with the `wire_to()` function of the `Node`
class -- using this function also makes the destination node aware
of the cable that connects to it. Let us say we have two state nodes `a` and `b`
and we wanted to wire them together such that the information in the `z`
compartment of `a` is transformed along a dense cable and finally deposited
into the `dz_td` compartment of state node `b`. This could be done with
the following code snippet (place the code in a script named `test_cable.py`):

```python
import tensorflow as tf
import numpy as np
from ngclearn.engine.nodes.snode import SNode

# create the initialization scheme (kernel) of the dense cable
init_kernels = {"A_init" : ("gaussian",0.025)}
dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : 69}

# note that the dim of a does NOT have to equal that of b if using a dense cable
a = SNode(name="a", dim=5, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=5, beta=1, leak=0.0, act_fx="identity")
a_b = a.wire_to(b, src_comp="z", dest_comp="dz_td", cable_kernel=dcable_cfg) # wire a to b

print("Cable {} of type *{}* transmits from:".format(a_b.name, a_b.cable_type))
print("Node {}.{}".format(a_b.src_node.name, a_b.src_comp))
print(" to ")
print("Node {}.{}".format(a_b.dest_node.name, a_b.dest_comp))
```

which would print to your terminal the following:

```console
Cable a-to-b_dense of type *dense* transmits from:
Node a.z
 to
Node b.dz_td
```

Graphically, the above 2-node circut would look like what is depicted in the
figure below.

```{eval-rst}
.. table::
   :align: center

   +-------------------------------------------------------+
   | .. image:: ../images/tutorials/lesson1/2n_circuit.png |
   |   :scale: 75%                                         |
   |   :align: center                                      |
   +-------------------------------------------------------+
```

Note that cables can auto-generate their own `.name` based on the source and
destination node that they wire to (in the case above, the cable `a_b` would
auto-generate the name `a-to-b_dense`). If you want the cable that wires `a` to
`b` to be named something specific, you set the extra argument `name` in `wire_to()`
to the desired string and force that cable to take on the name you wish (make
sure you choose a unique name). Furthermore, note that a `DCable` has two
learnable synaptic objects you can trigger depending on how you initialize the
cable:
1) a matrix `A` representing the bundle of synaptic connections that will
be used to transform the source node of the cable and relay this information to
the destination node of the cable, and
2) a bias vector `b` representing the shift added to the transformed output signal
of the cable.

What, then, does the above `a_b` dense cable do mathematically? Let us label
`z` compartment of node `a` as $\mathbf{z}^a$ and the `dz_td` of node `b`
as $\mathbf{dz}^b_{td}$. Given this labeling, the dense cable will perform
the following transformation:

$$
\mathbf{s}_{out} = \mathbf{z}^a \cdot \mathbf{A}^{a\_b} \\
\mathbf{dz}^b_{td} = \mathbf{dz}^b_{td} + \mathbf{s}_{out}
$$

where $\cdot$ denotes a matrix/vector multiplication and $\mathbf{A}^{a\_b}$ is the
matrix containing the synapses connecting the compartment `z` of node `a` to the
`dz_td` compartment of node `b`. If we had initalized the `DCable` earlier to have
a bias, like so:

```python
init_kernels = {"A_init" : ("gaussian",0.025), "b_init" : ("zeros")}
```

then the cable `a_b` would perform the following:

$$
\mathbf{s}_{out} = \mathbf{z}^a \cdot \mathbf{A}^{a\_b} + \mathbf{b}^{a\_b} \\
\mathbf{dz}^b_{td} = \mathbf{dz}^b_{td} + \mathbf{s}_{out}
$$

Notice that the last line in the above two equations also shows what each cable
will ultimately to node `b` -- they add in their transformed signal $\mathbf{s}_{out}$
to its $\mathbf{dz}^b_{td}$ compartment.

If you want to verify that the cable you wired from `a` to `b` appears
within node `b`'s `.connected_cables` data member, you can add/write a print
statement as follows:

```python
print("Cables that connect to Node {}:".format(b.name))
for cable in b.connected_cables:
    print(" => Cable:  {}".format(cable.name))
```

which would print to the terminal:

```console
Cables that connect to Node b:
 => Cable:  a-to-b_dense
```

Note that nodes `a` and `b` do not have to have the same `.dim` values if you
are wiring them together with a dense cable. In addition, cables in ngc-learn
are directional -- if you wire node `a` to node `b`, this does NOT mean that
node `b` is wired to node `a` (you would have to call the `wire_to()` funciton
again and create such a wire if this relationship is desired).

If you wanted to wire information directly from node `a` to node `b` WITHOUT
transforming the information via synapses, you can use a simple cable but, in
order to do so, the `.dim` data member (the number of neurons) of `a` must be
equal to that of `b`. You could write the following code (in a script you
name `test_cable2.py`):

```python
import tensorflow as tf
import numpy as np
from ngclearn.engine.nodes.snode import SNode

# create the initialization scheme (kernel) of the simple cable
scable_cfg = {"type": "simple", "coeff": 1.0}

## Note that you could do the exact same thing with a dense cable using
##   the two lines below but you would be wasting a matrix multiplication if so
# init_kernels = {"A_init" : ("diagonal",1)}
# dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : 69}

# note that the dim of a MUST be equal to b if using a simple cable
a = SNode(name="a", dim=5, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=5, beta=1, leak=0.0, act_fx="identity")
a_b = a.wire_to(b, src_comp="z", dest_comp="dz_td", cable_kernel=scable_cfg) # wire a to b

print("Cable {} of type *{}* transmits from:".format(a_b.name, a_b.cable_type))
print("Node {}.{}".format(a_b.src_node.name, a_b.src_comp))
print(" to ")
print("Node {}.{}".format(a_b.dest_node.name, a_b.dest_comp))
```

which would print to your terminal the following:

```console
Cable a-to-b_simple of type *simple* transmits from:
Node a.z
 to
Node b.dz_td
```

Wiring nodes with cables using the `.wire_to()` routine notably returns the
cable that it creates (in our code snippet this was stored in the variable `a_b`).
This is particularly useful if you need/want to set other properties of the generated
cable object such as local Hebbian synaptic update rules, constraints to be applied
to the cable's synapses, or synaptic value decay.

## Building Circuits with Nodes with Cables

Once you have created a set of nodes and wired them together in some meaningful
fashion, your circuit is now ready to be simulated.
To make the circuit function as a complete NGC dynamical system, you must place
your nodes into ngc-learn's simulation object, i.e., the `NGCGraph`. This object
will, once you have initialized it and made it aware of the nodes you want to
simulate, run some basic checks for coherence, internally configure the computations
that will drive the simulation that can leverage Tensorflow 2 static graph
optimization (you can turn this off if you do not want this optimization to happen),
and trigger the compilation routines inherent to each node and cable.

Specifically, if you wanted to compile the simple circuit you created in the last
section into a simulated NGC graph, you would then need to write the following
(you could add the following lines of code to your `test_cable.py` or `test_cable2.py`
scripts if you like to test the compile routine):

```python
from ngclearn.engine.ngc_graph import NGCGraph

circuit = NGCGraph()
circuit.set_cycle(nodes=[a,b]) # make the graph aware of nodes a and b, in that order
circuit.compile(batch_size=1)
```

where we see that the graph `circuit` is made aware of nodes `a` and `b` through
the call to `.set_cycle()` which takes in as argument a list of `Node` objects.
Notice that we do not have to explicitly tell the `NGCGraph` about the cable `a_b`
we created -- the `NGCGraph` will automatically handle the cable `a_b` through
the `.connected_cables` data member of all nodes it is made aware. The
`.compile()` routine will desirably do most of the heavy-lifting without much
input from the user except for a few small arguments if desired. For example,
in the code snippet above, we set the `batch_size` argument directly (the default
for an `NGCGraph` if you do not set it is also `1`), which is needed for the
default static graph optimization that the `NGCGraph` will set up after you call
`.compile()` -- note this also means you must make sure that the (mini-)batch size
of all sensory inputs you provide to the `NGCGraph` are the of length `batch_size`
(since ngc-learn makes use of in-place memory operators to speed up simulation
and play nicely with Tensorflow's static graph functionality).

If you do not wish to use the default static graph optimization and be able to
deal with variable-length mini-batches of data, then you can replace the above
call to `.compile()` by setting its `use_graph_optim` argument to `False` (which
has the trade-off that your simulations being slower).

Note that you can always "re-compile" an `NGCGraph` anytime you want. For example,
you wish to use the static graph optimization to speed up the training of your
`NGCGraph` circuit (since that is the most expensive part of simulating a stateful
neural system) but would like to reuse the trained graph on some new pool of
data samples with a different mini-batch size (or even, say, online, where you
feed in samples to the circuit one at a time).
You would simply write the code snippet exactly as we did earlier, run your
simulation of the training process, and then, after your code decides that
training is done, you could then simply re-compile your simulation object
to be dynamic (switching to Tensorflow eager execution mode) as follows:

```python
circuit.compile(use_graph_optim=False) # re-compile "circuit" to work w/ dynamic batch sizes
```

and you can then present inputs to your simulation object of any batch size you wish.
Alternatively, if you still wanted the benefit of the speed offered by static graph
optimization but just want to change the batch size to something different than what
was used during training (say you have a test set you want to sample mini-batches
of `128` samples instead), then you would write the following line:

```python
## NOTE: you can also re-compile your circuit to become a system with the same synaptic
## parameters but static-graph optimized for a different fixed batch size (w/o speed loss)
circuit.compile(batch_size=128) # <-- note all future batches of data must be length 128
```

re-compiling (as in the above two cases) provides some flexibility to the
experimenter/developer although a small setup cost is paid each the `.compile()`
routine is called.

Also, it is important to be aware that the `NGCGraph` itself internally maintains
several data structures that help it keep track of the simulated nodes/cables,
allow it to compute any desired synaptic updates, and ensure that the internal
dynamics interact properly with Tensorflow's static graph optimization while
still providing inspectability for the experimenter among other activities.
One particular object that will be of interest to you, the experimenter, is the
`.theta` list, which is the implementation of the mathematical construct $\Theta$
often stated in statistical learning and applied mathematics that houses ALL of
the learnable parameters (currently it would be empty in our case above because
we have not set any learning rules as we will later).

Given the above `NGCGraph`, you have now built your first, very own
custom NGC system. All that remains is to learn how to use an NGC system to process
some data, which we will demonstrate in the next section.

## Simulating an NGC Circuit with Sensory Data

In this section, we will illustrate two ways in which one may have an `NGCGraph`
interact with sensory data patterns.
Let us start by building a simple 3-node circuit, i.e., the one
depicted in the figure below (only the relevant compartments in each node
that we will wire together are depicted).

```{eval-rst}
.. table::
   :align: center

   +-------------------------------------------------------+
   | .. image:: ../images/tutorials/lesson1/3n_circuit.png |
   |   :scale: 75%                                         |
   |   :align: center                                      |
   +-------------------------------------------------------+
```

Create a Python file/script named `circuit1.py` and write the following to
create the header:

```python
import tensorflow as tf
import numpy as np

# import building blocks
from ngclearn.engine.nodes.snode import SNode
# import simulation object
from ngclearn.engine.ngc_graph import NGCGraph
```

Now write the following code for your circuit:

```python
integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}
a = SNode(name="a", dim=1, beta=1, leak=0.0, act_fx="identity",
          integrate_kernel=integrate_cfg)
b = SNode(name="b", dim=1, beta=1, leak=0.0, act_fx="identity",
          integrate_kernel=integrate_cfg)
c = SNode(name="c", dim=1, beta=1, leak=0.0, act_fx="identity",
        integrate_kernel=integrate_cfg)

init_kernels = {"A_init" : ("diagonal",1)}
dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : 69}
a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
c_b = c.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)

circuit = NGCGraph(K=5)
# execute nodes in order: a, c, then b
circuit.set_cycle(nodes=[a,c,b])
circuit.compile(batch_size=1)

# do something with the circuit above
a_val = tf.ones([1, circuit.getNode("a").dim]) # create sensory data point *a_val*
c_val = tf.ones([1, circuit.getNode("c").dim]) # create sensory data point *c_val*
readouts, _ = circuit.settle(
                clamped_vars=[("a","z",a_val), ("c","z",c_val)],
                readout_vars=[("b","phi(z)")]
              )
b_val = readouts[0][2]
print(" => Value of b.phi(z) = {}".format(b_val.numpy()))
print("             Expected = [[10]]")
circuit.clear()
```

The above (fixed) circuit will simply take in the current values within the `phi(z)` compartment
of nodes `a` and `c` and combine them together (through addition) within the
`dz_td` compartment of node `b`. Specifically, the value within the `phi(z)` compartment
of `a` will be transformed with the dense cable `a_b` and deposited first into
`dz_td` of `b` and then the compartment `phi(z)` of `c` will be transformed by the
dense cable `c_b` and added to the current value of and deposited into the
`dz_td` compartment of `b`.
Notice that we set the graph to execute the nodes in a particular order:
`a, c, b` so that way we ensure that first the values within nodes `a` and `c`
are first computed at any time step followed by node `b` which will then take
the current compartment values it needs from `a` and `c` and aggregate them
to compute its new state.
Alternatively, you could write and set up the same exact computation by organizing
the node computation into two subsequent cycles as follows:

```python
circuit = NGCGraph(K=5)
# execute nodes in order: a, c, then b
circuit.set_cycle(nodes=[a,c])
circuit.set_cycle(nodes=[b])
circuit.compile(batch_size=1)
```

where the above code-snippet is more explicit and, internally within the `NGCGraph`
simulation object, means that a separate computation cycle will be created that must  
wait on the first cycle (`a` then `b`) to be completed before it can then be executed
(note that the overall simulation needed for both would be the same when finally
run).

Now go ahead and run your `circuit1.py` (i.e., `$ python circuit1.py`) and you
should get the exact following output in your terminal:

```console
=> Value of b.phi(z) = [[10.]]
            Expected = [[10.]]
```

The above output should make sense since we clamped to the `phi(z)` compartments
of nodes `a` and `c` vectors of ones, after we run the `NGCGraph` for `K = 5`
steps of simulation time within the call to `.settle()`, we should obtain a vector
with `10` inside of it for the `phi(z)` compartment of node `b`. This is because,
at each time step within the `.settle()` function, the `dz_td` compartment of
node `b` is computed according to the following equation:

$$
\frac{\partial \mathbf{z}^b}{\partial t} &= \phi(\mathbf{z}^a) \cdot \mathbf{A}^{a\_b}
+ \phi(\mathbf{z}^c) \cdot \mathbf{A}^{c\_b} \\
 &= 1 \cdot \mathbf{A}^{a\_b} + 1 \cdot \mathbf{A}^{c\_b} = 1 \cdot \mathbf{I} + 1 \cdot \mathbf{I} \\
 & = (1 \cdot 1) + (1 \cdot 1) = 2
$$
<!--
```
b.dz_td = (a.phi(z) * a_b.A) + (c.phi(z) * c_b.A)
        = (1 * a_b.A) + (1 * c_b.A) = (1 * I) + (1 * I)
        = (1 * 1) + (1 * 1) = 2
```
-->

where $\mathbf{I}$ is the identity matrix (or diagonal matrix) of size `(1,1)` which is the
same as the scalar `1` (because we set the initialize of the `A` matrix within
cables `a_b` and `c_b` to be the diagonal matrix). This means that at any time
step, nodes `a` and `b` are combined ultimately depositing a scalar value of `2` into
node `b`'s `dz_td` compartment, which will then be added according to `b`'s
state dynamics:
$\mathbf{z}^b \leftarrow \mathbf{z}^b + \beta (\mathbf{dz}^b_{bu} + \mathbf{dz}^b_{td}) = \mathbf{z}^b + \beta (0 + \mathbf{dz}^b_{td}) $.
<!--`b.z = b.z + (dz_bu + dz_td) * beta = b.z + (0 + dz_td) * beta`.-->
If this calculation is repeated five times, as we have set the `NGCGraph` to do
via the argument `K=5`, then the circuit above is effectively repeatedly adding
`2` to the `z` compartment of node `b` five times (`2 * 5 = 10`). Note that for node
`b`, `phi(z)` is identical to the value of `z` because we set the activation function
of node `b` to be $\phi(\mathbf{z}) = \mathbf{z}$ or `act_fx = identity` (in fact, we have done this for all three
nodes in this example).

Now, let us slightly modify the above 3-node circuit code to go one step below the
application programming interface (API) of the `.settle()` and write our own
explicit step-by-step simulation so that way we can examine the value of the `z` and
`phi(z)` compartments of node `b` to prove that we are indeed accumulating a value
of `2` each time step.
To write a low-level custom simulation loop that does the same thing as the code
snippet we wrote earlier, you could replace the call to `.settle()` with the
following code instead:

```python
# ... same initialization code as before ...

# do something with the circuit above
a_val = tf.ones([1, circuit.getNode("a").dim])
c_val = tf.ones([1, circuit.getNode("c").dim])

circuit.clamp([("a","z",a_val), ("c","z",c_val)])
circuit.set_to_resting_state()
for k in range(K):
    values, _ = circuit.step(calc_delta=False)
    circuit.parse_node_values(values)
    b_val = circuit.extract("b","z")
    print(" t({}) => Value of b.phi(z) = {}".format(k, b_val.numpy()))
print("                  Expected = [[10.]]")
circuit.clear()
```

which will now print out to the terminal:

```console
t(0) => Value of b.phi(z) = [[2.]]
t(1) => Value of b.phi(z) = [[4.]]
t(2) => Value of b.phi(z) = [[6.]]
t(3) => Value of b.phi(z) = [[8.]]
t(4) => Value of b.phi(z) = [[10.]]
                 Expected = [[10.]]
```

showing us that, indeed, this circuit is incrementing the current value of the
`z` compartment by `2` each time step. The advantage to the above form of
simulating the stimulus window for the 3-node instead of using `.settle()` is that
one can now explicitly simulate the NGC system online if needed. This lower-level
way of simulating an NGC system would be desirable for very long simulation windows
where events might happen that interrupt or alter the settling process.

One final item to notice is that, in all of the code-snippets of this section, after
the `NGCGraph` has been simulated (either through `.settle()` or online via `.step()`),
we call the simulation objects `.clear()` routine. This is absolutely critical to
do after you simulate your NGC system for a fixed window of time *IF* you do not
want the current values of its internal nodes to carry over to the next time that
you simulate the system with `.settle()` or `.step()`. Since an NGC system is
stateful, if you expect its internal neural activities to have gone back to their
resting states (typically zero vectors) before processing a new pattern or batch
of data, then you must make sure that you call `.clear()`.
A typical design pattern for an NGC system would something like:

```python
# ... initialize the circuit and your optimizer *opt* earlier ...

# after sampling some data, a typical process loop would be:
readouts, delta = circuit.settle( ... ) # conduct iterative inference
opt.apply_gradients(zip(delta, circuit.theta)) # update synapses
circuit.clear() # set all nodes in system back to their resting states
```

## Evolving a Circuit over Time

### Shared/Linked Cables

While cables are intended to be unique in that they instantiate a particular
bundle of synapses that relay the information from one node to another, it is
sometimes desirable to allow two or more cables to reuse the exact same synapse
(pointing to the same spot in memory -- in other words, they make use of a shallow
copy of the synapses). This can also be useful if one needs to reduce the memory
footprint of their NGC system, e.g., for CPUs/GPUs with limited memory.
To facilitate sharing, you will need to use the `mirror_path_kernel` argument
of the `wire_to()` function you used earlier (in place of the `cable_kernel`
argument). This argument takes in a 2-tuple where the first argument is the
literal cable object you want to share parameters with and the second argument
is a string code/flag that tells ngc-learn which parameters (and how) to share.

In ngc-learn, one can make two cables "share" a bundle of synapses, and even bias
parameters, as follows (create a file called `circuit2.py` to place the
following code into):

```python
import tensorflow as tf
import numpy as np

# import building blocks
from ngclearn.engine.nodes.snode import SNode
# import simulation object
from ngclearn.engine.ngc_graph import NGCGraph

# create some nodes
a = SNode(name="a", dim=1, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=1, beta=1, leak=0.0, act_fx="identity")
x = SNode(name="x", dim=1, beta=1, leak=0.0, act_fx="identity")
y = SNode(name="y", dim=1, beta=1, leak=0.0, act_fx="identity")

init_kernels = {"A_init" : ("gaussian",0.1)}
dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : 111}

a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
# make cable *x_y* reuse the *A* matrix contained in cable *a_b*
x_y = x.wire_to(y, src_comp="phi(z)", dest_comp="dz_td", mirror_path_kernel=(a_b,"A"))

print("Cable {} w/ synapse A = {}".format(a_b.name, a_b.params["A"].numpy()))
print("Cable {} w/ synapse A = {}".format(x_y.name, x_y.params["A"].numpy()))
```

and you should see printed to your terminal:

```console
Cable a-to-b_dense w/ synapse A = [[0.1918097]]
Cable x-to-y_dense w/ synapse A = [[0.1918097]]
```

where we see that the cables `a_b` and `x_y` do indeed have the exact same
synaptic matrix of size `1 x 1` even though the cables themselves are completely
different and even connect completely different nodes (note that you would need
to make sure the `.dim` of node `x` is identical to node `a` and that the `.dim`
of node `y` is the same as node `b`, otherwise, you will get a shaping error
when the cable is later simulated).

There are other ways to share/point to synapses besides the direct way above.
For example, the code below will force cable `b_a` to reuse the transpose of
the `A` synaptic matrix of cable `a_b`, as indicated by the second code/flag `A^T`
input to the `mirror_path_kernel` argument:


```python
a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
# make cable *b_a* that reuses the transpose of the *A* matrix contained in cable *a_b*
b_a = b.wire_to(a, src_comp="phi(z)", dest_comp="dz_td", mirror_path_kernel=(a_b,"A^T"))
```

Other useful codes for the `mirror_path_kernel` argument include:
`A+b` which shares the `A` matrix and bias `b` of the target cable and
`-A^T` which shares the negative transpose of matrix `A` of the target cable.

### Synaptic Update Rules

A key element of an NGC system is its ability to evolve with time and learn
from the data patterns it processes by updating its synaptic weights.
To update the synaptic bundles (and/or biases) inside the cables you use to
wire together nodes, you will need to also define corresponding learning rules.
Currently, ngc-learn assumes that synapses are adjusted through locally-defined
multi-factor Hebbian rules.

To configure a cable, particularly a dense cable, to utilize an update rule,
you need to specify the following with the `set_update_rule()` routine:

```python
a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
a_b.set_update_rule(preact=(a,"phi(z)"), postact=(b,"phi(z)"), param=["A"])
```

where we must define at least three arguments:
1) the pre-activation term `preact` which must be a 2-tuple containing the
pre-activation node object and a string stating the compartment that we want
to extract a vector signal from,
2) the post-activation term `postact` defined exactly the same as the pre-activation
term, and
3) a list of strings `param` stating the synaptic parameters we want the update rule
to affect.
The code-snippet above will tell ngc-learn that when cable `a_b` is updated, we
would like to take the (matrix) product of node `a`'s `phi(z)` compartment and
node `b`'s `phi(z)` compartment and specifically adjust matrix `A` within the cable.

If cable `a_b` also contained a bias, we would specify the rule as follows:

```python
a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
a_b.set_update_rule(preact=(a,"phi(z)"), postact=(b,"phi(z)"), param=["A", "b"])
```

and ngc-learn will intelligently realize that synaptic vector `b` of cable `a_b`
will be updated using only the post-activation term `postact` (since it is a
vector and not a matrix like `A`).

Using the `.set_update_rule()` function on each cable that you would like to evolve
or be updated given data is all that you need to do to set up local learning. The
`NGCGraph` will automatically become aware of the valid cables linking
nodes that are learnable and, internally, call those cables' update rules to
compute the correct synaptic adjustments. In particular, whenever you call
`.settle()` on an `NGCGraph`, the simulation object
will actually compute ALL of the synaptic adjustments at the end of the simulation
window and store them into a list `delta` and return them to you.

For example, you want to compute the Hebbian update for the cable `a_b` earlier
(that you wrote for `circuit2.py`) given a data point containing the value of
one (create a new file and write the code below into `circuit3.py`):

```python
import tensorflow as tf
import numpy as np

from ngclearn.engine.nodes.snode import SNode # import building blocks
from ngclearn.engine.ngc_graph import NGCGraph # import simulation object

# create the initialization scheme (kernel) of the dense cable
init_kernels = {"A_init" : ("gaussian",0.1)}
dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : 111}

a = SNode(name="a", dim=1, beta=1, leak=0.0, act_fx="identity")
b = SNode(name="b", dim=1, beta=1, leak=0.0, act_fx="identity")
a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
a_b.set_update_rule(preact=(a,"phi(z)"), postact=(b,"phi(z)"), param=["A"])

print("Cable {} w/ synapse A = {}".format(a_b.name, a_b.params["A"].numpy()))

circuit = NGCGraph()
# execute nodes in order: a, c, then b
circuit.set_cycle(nodes=[a,b])
circuit.compile(batch_size=1)

opt = tf.keras.optimizers.SGD(0.01)

# do something with the circuit above
a_val = tf.ones([1, circuit.getNode("a").dim]) # create sensory data point *a_val*
readouts, delta = circuit.settle(
                    clamped_vars=[("a","z",a_val)],
                    readout_vars=[("b","phi(z)")]
                  )
opt.apply_gradients(zip(delta, circuit.theta))
circuit.clear()

print("Update to cable {} is: {}".format(a_b.name, delta[0].numpy()))
```

which would print to your terminal:

```console
Update to cable a-to-b_dense is: [[-0.9590485]]
```

Notice that we have demonstrated how ngc-learn interacts with Tensorflow 2
optimizers by simply giving the returned `delta` list and the circuit's internal
`.theta` list to the optimizer which will then physically adjust the values of
synaptic bundles themselves for you. NOTE that the order of Hebbian updates will be
returned in the exact same order as the learnable parameters that `.theta` points to.

The above NGC system is, of course, rather naive as we would effectively be calculating
and update the single synapses that connects nodes `a` and `b`, and, since this
use of the update rule is classical Hebbian, the value of the synapse inside of `A`
of cable `a_b` would grow indefinitely.
In the next section, we will craft a more interesting circuit that uses what you
learned about with respect cables and nodes, including the error node `ENode`.

## Constructing a Convergent 5-Node Circuit

As our final exercise for this tutorial, let us build a 4-node circuit that
attempts to learn how to converge to a state such that a five-dimensional node `a`
and a six-dimensional node `b` each generate three-dimensional output values  
that are nearly identical. In other words, we want node `a` to get good at
predicting the output of node `b` and node `b` to get good at predicting the
output of node `a`. Furthermore, node `b`'s `z` compartment will always be clamped
to a vector of ones.
To measure the mismatch between these two nodes' predictions, we will introduce
the fifth and final node as a three-dimensional error node tasked with
computing how far off the two sources nodes are from each other.

We illustrate the 5-node circuit in the figure below. The relevant compartments
that we will be wiring together are shown as different-colored circles (and the
legend maps the color to the compartment name).

```{eval-rst}
.. table::
   :align: center

   +-------------------------------------------------------+
   | .. image:: ../images/tutorials/lesson1/5n_circuit.png |
   |   :scale: 75%                                         |
   |   :align: center                                      |
   +-------------------------------------------------------+
```

To build this circuit, create a file called `circuit4.py` and write the header:

```python
import tensorflow as tf
import numpy as np

# import building blocks
from ngclearn.engine.nodes.enode import ENode
from ngclearn.engine.nodes.snode import SNode
# import simulation object
from ngclearn.engine.ngc_graph import NGCGraph
```

and then go ahead and create the 5-node circuit we described as follows:

```python
# create the initialization scheme (kernel) of the dense cable
init_kernels = {"A_init" : ("gaussian",0.1)}
dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : 111} # dense cable
scable_cfg = {"type": "simple", "coeff": 1.0} # identity cable

a_dim = 5
e_dim = 3
b_dim = 6

# Define node a
a = SNode(name="a", dim=a_dim, beta=1, leak=0.0, act_fx="identity")
# Define node a_mu
a_mu = SNode(name="a_mu", dim=e_dim, beta=1, zeta=0, leak=0.0, act_fx="identity")
# Define error node e
e = ENode(name="e", dim=e_dim)
# Define node b
b = SNode(name="b", dim=b_dim, beta=1, leak=0.0, act_fx="identity")
# Define node b_mu
b_mu = SNode(name="b_mu", dim=e_dim, beta=1, zeta=0, leak=0.0, act_fx="identity")

# wire a to a_mu
a_amu = a.wire_to(a_mu, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
a_amu.set_update_rule(preact=(a,"phi(z)"), postact=(e,"phi(z)"), param=["A"])
# wire a_mu to e
amu_e = a_mu.wire_to(e, src_comp="phi(z)", dest_comp="pred_mu", cable_kernel=scable_cfg)

# wire b to b_mu
b_bmu = b.wire_to(b_mu, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
b_bmu.set_update_rule(preact=(b,"phi(z)"), postact=(e,"phi(z)"), param=["A"])
# wire b_mu to e
bmu_e = b_mu.wire_to(e, src_comp="phi(z)", dest_comp="pred_targ", cable_kernel=scable_cfg)

# wire e back to a
e_a = e.wire_to(a, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(a_amu,"A^T"))

# wire e back to b
e_b = e.wire_to(b, src_comp="phi(z)", dest_comp="dz_bu", mirror_path_kernel=(b_bmu,"A^T"))

circuit = NGCGraph()
# execute nodes in order: a, c, then b
circuit.set_cycle(nodes=[a, a_mu, b, b_mu])
circuit.set_cycle(nodes=[e])
circuit.set_learning_order([b_bmu, a_amu]) # enforces order - b_bmu then a_amu
circuit.compile(batch_size=1)

opt = tf.keras.optimizers.SGD(0.05)
```

and then, given the 5-node graph you crafted and compiled above, you can
now write a simple training loop to simulate as in below:

```python

n_iter = 60 # number of overall optimization steps to take

b_val = tf.ones([1, circuit.getNode("b").dim]) # create sensory data point *b_val*
print("---- Simulating Circuit Evolution ----")
for t in range(n_iter):
    readouts, delta = circuit.settle(
                        clamped_vars=[("b", "z",b_val)],
                        readout_vars=[("e", "L")]
                      )
    e_val = readouts[0][2]
    if t > 0:
        print("\r{} => Value of e.L = {}".format(t, e_val.numpy()),end="")
    else:
        print("{} => Value of e.L = {}".format(t, e_val.numpy()))
    opt.apply_gradients(zip(delta, circuit.theta))
    circuit.clear()
print()

print("---- Final Results ----")
# get final values
readouts, delta = circuit.settle(
                    clamped_vars=[("b", "z",b_val)],
                    readout_vars=[("e", "pred_mu"),("e", "pred_targ")],
                    calc_delta=False # turn off update computation
                  )
prediction = readouts[0][2].numpy()
target = readouts[1][2].numpy()
print("Prediction: {}".format(prediction))
print("    Target: {}".format(target))
circuit.clear()
```

Once you have written `circuit4.py`, you can execute it from the command line
as `$ python circuit4.py` which should print to your terminal something similar to:

```console
---- Simulating Circuit Evolution ----
0 => Value of e.L = [[0.03118673]]
59 => Value of e.L = [[3.9547285e-05]]
---- Final Results ----
Prediction: [[-2.2072585  -1.1418786   0.68785524]]
    Target: [[-2.2125444 -1.1444474  0.6895305]]
```

As you can see, the loss represented by the error node `e` (specifically, the
value stored in its loss `L` compartment), starts at greater than `0.03` and
then decreases over the sixty simulated training iterations to nearly zero
(`0.00003954`), and, as we can see in the comparison between the prediction
from node `a` against the target produced by node `b`, the values are quite close.
This indicates that our small 5-node circuit has converged to an equilibrium point
where node `a` and node `b` are capable of matching each other (assuming that
`b`'s `z` compartment will always be clamped to a vector of ones). Furthermore,
we see that we have crafted a feedback loop via cable `e_a`, which transmits
the error information contains inside of node `e` back to the `dz_bu` compartment
of node `a`, which, as we recall from the earlier part of this tutorial, is used
in node `a`'s state update equation. (Feedback loop `e_b` does something similar
to `e_a`, however, since we force the `z` compartment of `b` to always be a
specific value, this loop ends up being useful in this example).

With the completion of the above example, you have now gone through the process
of crafting your own custom NGC circuit with ngc-learn's nodes-and-cables system.
Given this knowledge, you are ready to design and simulate your own predictive
processing neural systems based on the NGC computational framework. For examples
of how nodes and cables are used to build various classical and modern-day models,
check out the [Model Museum](../museum/model_museum.md) (including the
pre-designed agents in the ngc-learn repo `ngclearn/museum/`) and the walk-throughs.

## Knowing the Utility Functions of an NGCGraph

Although you have learned of and how to assemble the key elements in ngc-learn
needed to construct NGC circuits, there are a few useful utility functions that
are provided once you construct the `NGCGraph` simulation object. In this
closing section, we will briefly discuss each of these and briefly illustrate
their use (and review key ones that we covered earlier).

### Compiling and Re-Compiling Your Simulation Object

As discussed earlier in this tutorial lesson, the `.compile()` is one of the
most important functions to call after you have constructed your `NGCGraph` as it
will set up the crucial internal bookkeeping and checks to ensure that your
simulated NGC system works correctly with static graph optimization and is
properly analyzable.

Normally, just calling the `.compile()` function after you initialize the
`NGCGraph` constructor is sufficient so long as you either set its `batch_size`
argument to the batch size you will be training with (and you must ensure
that your data is presented to your graph in batch sizes with that exact
same length each time, otherwise the `NGCGraph` will throw a memory error). Note
that you can also set the batch size your graph expects in the constructor itself,
like so `NGCGraph(K=10, batch_size=128)`.

If you do not wish for ngc-learn to use static graph optimization, you can always
turn this by setting the `use_graph_optim` to `False` in the `.compile()` function,
which will allow you to use variable-length batch sizes (and not force you to
specify the `batch_size` in the compile routine or in the `NGCGraph` constructor)
but this will come at the cost of slower simulation time especially if you will
be evolving the synapses over time (only in the case of pure online learning might
turning off the static graph optimization be useful).
However, you can, as was discussed earlier, always "re-compile" your simulation
object if, for example, you will be training with mini-batches of one length
and then testing with mini-batches of another length. Re-compiling is simple and
not too expensive to do if done sparingly -- all you need to do is call
`.compile()` again and choose a new `batch_size` to give it as an argument.

One final note about the `.compile()` routine is that it actually returns a
dictionary of dictionaries that contains/organizes the core specifications
of your `NGCGraph`. You can print this dictionary out if you like and examine
that the various nodes and cables state the various key properties you expect them
to aid in debugging. Future plans for ngc-learn will be to leverage this
simulation properties dictionary to aid in auto-generated visualization to
help in creating architecture figures and possibly information-flow diagrams
(we would also like to mention here that we welcome
[community contributions](https://github.com/ago109/ngc-learn/blob/main/CONTRIBUTING.md)
with respect to visualization and system analysis if you are interested in helping
with this particular effort).

### Clearing the State of Your Simulator

Another core routine that you learned about in this tutorial is the `.clear()`
function. This is a critical function to call whenever you want to completely
wipe out the state of your `NGCGraph` simulation object. Wiping graph state is
something you will likely want to do quite often in your code. For example,
a typical design pattern for simulating an NGC system after you sample a batch of
training data points is to: 1) first call its `.settle()` function, 2) do something
with the readout variables you asked it to return (and maybe extract some other
items from your graph), 3) update the system's synaptic weights (as housed in
its `.theta` construct) using an external optimization algorithm like stochastic
gradient descent, 4) apply/enforce constraints, and 5) clear/wipe the graph state.

There are no arguments to `.clear()` but you should be aware that it does wipe
the state of your graph thoroughly -- this also means that, after clearing, using
a getter function `.extract()` (discussed in the next section) becomes meaningless
the internal bookkeeping structures that your graph maintains get set to their
default ("empty") states. Note that clearing the graph state is **NOT** the same
as setting nodes exactly to their resting state -- node resting states are actually
set with a call to `.set_to_resting_state()` and this is actually done for you
every time you call `.settle()` (unless you tell your graph not to start at a
resting state by setting the `cold_start` flag argument to `False`).

Note that a use-case where you might *not* want to use the `.clear()` function
is if you are simulating an NGC system over one long, single window of time (for
example, a sensory data stream). In this scenario, using `.clear()` would be
against the processing task as the neural system should be aware of its previous
nodal compartment activities after the last call to `.settle()` (you would want
to also set `cold_start` to `False` in this situation). We remark that a better
alternative to using `.settle()` for streaming data applications is to, like we
did early in this tutorial, work with the lower-level API of your `NGCGraph` and
just use its `.step()` routine which *exactly* simulates one discrete step of
time of your graph. This would allow you to set up "events" such as when you want
`.step()` to return updates to synapses (by setting the `calc_delta` argument to `True`
if you do and `False` otherwise) and when you want node compartments to go to
their actual resting states with a call to `.set_to_resting_state()`.
While it is flexible, we caution the user that leveraging the lower-level online
functionality of an `NGCGraph` does require some degree of comfort with how
ngc-learn operates and care should be taken to check that your system is evolving
in the way that you expect (working with the online functionality of an NGC
system will be the subject of a future advanced lesson).

### Extracting Signals and Properties: Getter Functions

Two of the most important "getter" functions you will want to be familiar with
when dealing with `NGCGraph`'s are `.extract()` and `.getNode()`.

The `.extract()` function is useful when you want to access particular values of your
NGC system at a particular instant. For example, let us say that you want to
retrieve and inspect the value of the `z` compartment of the node `a` in the
5-node circuit you built in the last section right. You would then utilize the
`.extract()` methods as follows:

```python
node_value = circuit.extract("a", "z")
print(" -> Inspecting value of Node a.z = {}".format(node_value.numpy()))
```

which would print to your terminal:

```console
 -> Inspecting value of Node a.z = [[-0.9275531   2.341278    0.2365013   1.2464949   0.76036114]]
```

NOTE: it is meaningless to call `.extract()` in the following two cases:
1) after you call `.clear()`, as `.clear()` will completely wipe the state of
your `NGCGraph`, and
2) not before you have simulated used your `NGCGraph` for any amount of time
(if you have never simulated the graph, then your graph has no signals of any
meaning since it has never interact with data or an environment).
If you call `.extract()` in cases like those above, it will simply return `None`.

The `.getNode()` is useful if you have already compiled your `NGCGraph` simulation
object and want to retrieve properties related to a particular node in this graph.
For example, let us say that you want to determine the dimensionality of the
`e` node in your 5-node circuit of the last section. To do this, you would
write the following code:

```python
node = circuit.getNode("e")
print(" -> The dimensionality of Node e is {}".format(node.dim))
```

which would print to your terminal:

```console
 -> The dimensionality of Node e is 3
```

The `.getNode()` method will return the full `Node` object of the same exact
name you input as argument. With this object, you can query and inspect any of
its internal data members, such as the `.connected_cables` as we did earlier in
this lesson.

### Clamping and Injecting Signals: Setter Functions

The two "setter" functions that will you find most useful when working with
the `NGCGraph` are `.clamp()` and `.inject()`. Clamping and injecting, which
both work very similarly, allow you to force certain compartments in certain
nodes of your choosing to take on certain values before you simulate the `NGCGraph`
for a certain period of time. While both of these initially place values into
compartments, there is a subtle yet important difference in the effect each has
on the graph over time. Desirably, both of these functions take in a list of
arguments, allowing you clamp or inject many items at one time if needed.

In the event that you want a particular node's compartment to take on a specific
set of values and **remain fixed at** these values throughout the duration of
a simulation time window, then you want to use `.clamp()`. In our 5-node circuit
earlier, we in fact did this in our particular call to `.settle()` (which, internally,
actually makes a call to `.clamp()` for you if you provide anything to the `clamped_vars`
argument), but you could, alternatively, use the clamping function explicitly if you
need to as follows:

```python
b_val = tf.ones([1, circuit.getNode("b").dim])
circuit.clamp([("b", "z", b_val)])
readouts, delta = circuit.settle(
                    readout_vars=[("e", "pred_mu"),("e", "pred_targ")],
                    calc_delta=False # turn off update computation
                  )

node_value = circuit.extract("b", "z")
print(" -> Inspecting value of Node b.z = {}".format(node_value.numpy()))
````

which will, through each step of simulation conducted within the `.settle()`
force the `z` compartment of node `b` to ALWAYS remain at the value of `b_val`
(this vector of ones will persist throughout the simulation time window).
The result of this code snippet prints to terminal the following:

```console
 -> Inspecting value of Node b.z = [[1. 1. 1. 1. 1. 1.]]
```

This is as we would expect -- we literally clamped a vector of six ones to `z` of
node `b` and would expect to observe that this is still the case at the end of
simulation.

If, in contrast, you only want to **initialize** a particular node's compartment
to *start* at a specific value but not necessarily remain at this value, you
will want to use `.inject()`. Doing so looks like code below:

```python
b_val = tf.ones([1, circuit.getNode("b").dim])
circuit.inject([("b", "z", b_val)])
readouts, delta = circuit.settle(
                    readout_vars=[("e", "pred_mu"),("e", "pred_targ")],
                    calc_delta=False # turn off update computation
                  )

node_value = circuit.extract("b", "z")
print(" -> Inspecting value of Node b.z = {}".format(node_value.numpy()))
````

which looks nearly identical to the clamping code we wrote above. However, the
result of this computation is quite different as seen in the terminal output
below:

```console
 -> Inspecting value of Node b.z = [[8.505673  8.249885  8.257135  7.7380524 8.38973   8.267948 ]]
```

Notice that the values within `z` of node `b` are NOT ones like we saw in our
previous clamping example. This is because this compartment only started at the
first time step as a vector of ones but, according to the internal dynamics of
node `b` which are driven by the originally useless feedback loop/cable
`e_b` we created earlier -- recall, at the time, that we wrote that this cable
would do nothing because we *clamped* `z` in node `b` to a vector of ones. If
we had instead *injected* the vector of ones, this compartment in node `b`
would indeed have evolved over time.

### Enforcing Constraints

One final item that you may find important when simulating the evolution of
an `NGCGraph` is the enforcing of constraints through the `.apply_constraints()`
routine.
For example, you want to ensure that the Euclidean norms of the columns of a
particular matrix `A` in one of your system's cables never exceed a certain value
(see [Walkthrough \#4](../walkthroughs/demo4_sparse_coding.md) for a case that requires
this constraint to be true).

To enforce a constraint on a particular cable, all you need to do is first make
the desired cable aware of this constraint like so:

```python
a_amu = a.wire_to(a_mu, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
constraint_cfg = {"clip_type":"norm_clip","clip_mag":1.0,"clip_axis":1}
a_amu.set_constraint(constraint_cfg)
```

then, whenever you call the `.apply_constraints()` of your `NGCGraph` simulation object,
this constraint will internally be enforced/applied to the cable `a_amu`. Typically,
this call looks like the following (using our 5-node circuit as an example):

```python
readouts, delta = circuit.settle(
                    clamped_vars=[("b", "z",b_val)],
                    readout_vars=[("e", "L")]
                  )
opt.apply_gradients(zip(delta, circuit.theta))
circuit.apply_constraints() # generally apply constraints after an optimizer is used...
circuit.clear()
```

where we see that we call `.apply_constraints()` AFTER the Tensorflow optimizer
has been used to actually alter the values of the synapses of the NGC system.
If, after the `SGD` update had resulted in the norms of any of the columns in
the matrix `A` of cable `a_amu` to exceed the value of `1.0`, then `.apply_constraints()`
would further alter this matrix to make sure it no longer violates this constraint.

**A Note on Synaptic Decay:** Like norm constraints, weight/synapse decay is also
treated as a (soft) constraint in an `NGCGraph`. If you want to apply a small
decay to a particular synaptic bundle matrix `A` in a particular cable, you can
easily do so by simply calling the `.set_decay()` function like so:

```python
a_b.set_decay(decay_kernel=("l1",0.00005)) # apply L1 weight decay to *A* in cable *a_b*
```

which would apply a decay factor based on a centered Laplacian distribution (
or an L1 penalty). If you chose `l2` instead, the decay factor applied would then
be based on a centered Gaussian distribution (or an L2 penalty) over each element in matrix `A`
of cable `a_b`.

## Conclusion

You now have successfully gone through the core details and functionality of ngc-learn's
nodes-and-cables system. The next step is to build your own NGC systems/models
for your own research projects and play with the pre-designed systems in the Model
Museum (and go through the walkthroughs). In future upcoming tutorial lessons, we will
cover topics such designing your own customs nodes or cables that interact with
the nodes-and-cables system and working with the low-level online functionality of
simulated NGC systems.

## References
Hebb, Donald Olding. The organization of behavior: A neuropsychological theory.
Psychology Press, 2005.
