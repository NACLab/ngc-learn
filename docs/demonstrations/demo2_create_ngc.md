# Demo 2: Creating Custom NGC Systems

<i>NOTE: This demonstration is under construction and thus incomplete at the moment...</i>

## Theoretical Motivation: Cables and Compartments
In this demonstration, we will learn how to craft our own custom NGC system
using ngc-learn's fundamental building blocks -- nodes and cables. At its core,
part of ngc-learn's fundamental design is inspired by (neural)
<a href="http://www.scholarpedia.org/article/Neuronal_cable_theory">cable theory </a>,
where neurons, arranged in complex connectivity structures, are viewed as
performing dendritic calculations. In other words, a particular neuron integrates
information from different input signals (for example, those from other neurons), in
often highly nonlinear ways through a complex dendritic tree.

Although modeling a neuronal system through the lens of cable theory is certainly
complex and intricate in of itself, ngc-learn is built in this direction, starting
with the idea a neuron (or a cluster of them) can be viewed as a node, or
[Node](ngclearn.engine.nodes.snode) (also see {ref}`node-model`), and each bundle of synapses that connect nodes can be viewed as a cable, or [Cable](ngclearn.engine.cables.cable) (also see {ref}`cable-model`).
Each node has different, multiple "compartments" (which we allow to be named, if desired),
which allows it to collect information from many different connected/related nodes
and then, within its integration routine (or `step()`), decide how to combine the
different signals in order to calculate its own activity (loosely corresponding to a
rate-coded firing rate -- we will learn how to model spike trains in a later
demonstration). As a result, many nodes and cables yield an NGC system where each
node is itself, in general, a stateful computation (even if we are processing static
data such as images).

## Building and Simulating NGC Systems

### The Building Blocks: Nodes and Cables
With the above aspect of ngc-learn's theoretical framing in mind, we can craft
connectivity patterns of our own by deciding the form that each node and cable
in our system will take. ngc-learn currently offers a few core nodes and cable types
(note ngc-learn is an evolving software framework, so more node/cable types are to come
in future releases, either through the NAC team or community contributions).
The core node type set currently includes `SNode`, `ENode`, and `FNode` (all inheriting
from the `Node` base class) while the current cable type set includes `DCable` and
`SCable` (all inherited from the `Cable` base class).

An `SNode` refers to a stateful node (see [SNode](ngclearn.engine.nodes.snode)),
which is one of the primary nodes you will
work with when crafting NGC systems. A stateful node contains inside of it a cluster
(or block) of neurons, the number of which is controlled through the `dim`
argument. To initialize a state node, we simply invoke the following:

```python
integrate_cfg = {"integrate_type" : "euler", "use_dfx" : True}
prior_cfg = {"prior_type" : "laplace", "lambda" : 0.001}
a = SNode(name="a", dim=64, beta=0.1, leak=0.001, act_fx="relu",
           integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
```

where we notice that the above creates a state node with `64` neurons that will
update themselves according to an Euler integration step (step size of `0.1` and
a leak of `0.001`) and apply a `relu` post-activation to compute their
post-activity values. Furthermore, notice that a Laplacian prior has been place
over the neural activities within the state `a` (weighted by the strength
coefficient `lambda`) -- such a prior is meant to encourage neural activity values
towards zero (yielding sparser patterns).
A state node, in ngc-learn 0.0.1, contains five key compartments: `dz_td`, `dz_bu`,
`z`, `phi(z)`, and `mask`. `z` represents the actual state values of the neurons
inside the node while `phi(z)` is the nonlinear transform of `z` (indicating the
application of the node's encoded activation/transfer function, e.g., `relu` in
the case of node `a` in the example above). `dz_td` and `dz_bu` are state update
compartments, where (vector) signals from other nodes are generally deposited
(and summed together vector-wise), with the notable exception that `dz_bu` is
weighted by the first derivative of the activation function encoded into the
state node (for example, in `a` above, signals deposited into `dz_bu` are
element-wise multiplied by the relu derivative, or `d.phi(z)/d.z = d.relu(z)/d.z`).
While, in principle, any node can deposit into any compartment of a state node,
the intentional and primary use of an `SNode` entails letting the node itself
automatically update `z` and `phi(z)` according to the integration function set
(such as Euler integration) and having other nodes deposit activity values in
`dz_td` and `dz_bu`. (This demonstration will assume this form of operation.)

While a state node by itself is not all that interesting, when we connect it to
another node, we create a basic computation system where signals are passed from
a source node to a destination node. To connect a node to another node, we need
to wire them together with a `Cable`, which can transform signals between them
with a dense bundle of synapses (as in the case of a `DCable`) or simply carry
along and potentially weight by a fixed scalar multiplication (as in the case of
an `SCable`). For example, if we want to wire node `a` to a node `b` through a
dense bundle of synapses, we would do the following:

```python
a = SNode(name="a", dim=64, beta=0.1, leak=0.001, act_fx="relu",
          integrate_kernel=integrate_cfg, prior_kernel=prior_cfg)
b = SNode(name="b", dim=32, beta=0.05, leak=0.002, act_fx="identity",
          integrate_kernel=integrate_cfg, prior_kernel=None)

dcable_cfg = {"type": "dense", "has_bias": False,
              "init" : ("gaussian",0.025), "seed" : 69}
a_b = a.wire_to(b, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
```

where we note that the cable/wire `a_b`, of type `DCable` (see [DCable](ngclearn.engine.cables.dcable)),
will pull a signal from the `phi(z)` compartment of node `a` and transmit/transform
this signal along the synaptic parameters it embodies (a dense matrix where each synaptic
value is randomly initialized from a zero-mean Gaussian distribution and
standard deviation of `0.025`) and place the resultant signal inside
the `dz_td` compartment of `b`.

Currently, an `SNode` (in ngc-learn version 0.0.1), integrates over two
compartments -- `dz_td` (top-down pressure signals) and `dz_bu` (bottom-up
potentially weighted signals), and finally combines them through a linear combination
to produce a full update to the internal state compartment `z`. Note that many
external nodes can deposit signal values into each compartment `dz_td` and `dz_bu`
and each new deposit value is directly summed with the current value of the compartment.
For example, a five-node system or circuit could be as follows:

```python
carryover_cable_cfg = {"type": "simple", "coeff": 1.0}
a = SNode(name="a", dim=10, beta=0.1, leak=0.001, act_fx="identity")
b = SNode(name="b", dim=5, beta=0.05, leak=0.002, act_fx="identity")
c = SNode(name="c", dim=2, beta=0.05, leak=0.0, act_fx="identity")
d = SNode(name="d", dim=2, beta=0.05, leak=0.0, act_fx="identity")
e = SNode(name="e", dim=15, beta=0.05, leak=0.0, act_fx="identity")

a_c = a.wire_to(c, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
b_c = b.wire_to(c, src_var="phi(z)", dest_var="dz_td", cable_kernel=dcable_cfg)
d_c = d.wire_to(c, src_var="phi(z)", dest_var="dz_bu", cable_kernel=carryover_cable_cfg)
e_c = e.wire_to(c, src_var="phi(z)", dest_var="dz_bu", cable_kernel=dcable_cfg)
```

where `a` and `b` both deposit signals (which will be summed) into the `dz_td`
compartment of `c` while `d` and `e` deposit signals into the `dz_bu`
compartment of `c`. Crucially notice that we introduce the other type of cable from
`d` to `c`, i.e., an `SCable` (see [SCable](ngclearn.engine.cables.scable)), which
is a simple carry-over cable that we have merely
configured (in the dictionary `carryover_cable_cfg`) to only pass along information
information from node `d` to `c`, simply multiplying the vector by `1.0` (NOTE:
if a simple cable is being used, the dimensionality of the source node and the
destination node should be exactly the same).
Bear in mind that any general `Cable` is directional -- it only
transmits in the direction of its set wiring pattern. So if it is desired, for
instance, that information flows not only from `a` to `c` but from `c` to `a`,
then one would need to directly wire node `c` back to `a` following a similar
pattern as in the code snippet above. Finally, note that when you wire together
two nodes, they each become aware of this wiring relationship (i.e., node `a`
understands that it feeds into node `c` and node `c` knows that `a` feeds into it).

To learn or adjust the synaptic cables connecting the nodes in the five-node
system we created above, we need to configure the cables themselves to use
a local Hebbian-like update. For example, if we want the cable `a_c` to evolve
over time, we notify it that it needs to update according to:

```python
a_c.set_update_rule(preact=(a,"phi(z)"), postact=(c,"phi(z)"))
```

where the above sets a (two-factor) Hebbian update that will compute an update/adjustment
matrix of the same shape as the underlying synaptic matrix that connects `a` to
`c` (essentially a product of post-activation values in `a` with post-activation
values in `c`). Notice that a pre-activation term (`preact`) requires a 2-tuple
containing a target node object and a string denoting which compartment within
that node to extract information from to create the pre-synaptic Hebbian term.
(`postact` refers to the post-activation term, the argument itself following the
same format at `preact`).

In addition to the `SNode`, we need to turn our attention to one more important
type of node -- the `ENode` (see [ENode](ngclearn.engine.nodes.enode)). While,
in principle, one could build a complete NGC system with just state nodes and
cables (which will be the subject of future
demonstrations/tutorials), an important aspect of NGC computation we have not
addressed is that of the `error neuron`, represented in ngc-learn by an `ENode`.
An `ENode` is a special type of node that performs a mismatch calculation (or a
computation that compares how far off one quantity is from another) and is, in
fact, a mathematical simplification of a state node known as a fixed-point.
In short, one can simulate a mismatch calculation over time by simply modeling
the final result such as the (vector) subtraction of one value from another. In
ngc-learn version 0.0.1, in addition to `z` and `phi(z)`, the `ENode` contains
the following compartments: `pred_mu`, `pred_targ`,
`L`, and `avg_scalar`. `pred_mu` is a compartment that contains a summation
of deposits that represent an external signals that form a "prediction" (or expectation)
while `pred_targ` is a compartment that contains a summation of external signals
that form a "target" (or desired value/signal). `L` is a useful compartment as
this is internally calculated by the error node as it is the loss function by which
the fixed-point calculation is derived, i.e., in the case of simple subtraction or
`pred_mu - pred_targ`, this would mean that the error node is calculating the first
derivative of the mean squared error (MSE). `avg_scalar` is a special signal
that can be set externally and is to contain a scalar that each value inside of
`z` and `L` will be multiplied by `1.0/avg_scalar`.

Now that we know how an error node works, let us modify the earlier 5-node circuit
to do something more interesting with a mismatch computation:

```python
a = SNode(name="a", dim=10, beta=0.05, leak=0.001, act_fx="identity")
b = SNode(name="b", dim=10, beta=0.05, leak=0.002, act_fx="identity")
e = ENode(name="e", dim=5)
c = SNode(name="c", dim=10, beta=0.05, leak=0.0, act_fx="identity")
d = SNode(name="d", dim=10, beta=0.05, leak=0.0, act_fx="identity")

# wire the states a, b, c, and d to error neurons/node e
a_e = a.wire_to(e, src_var="phi(z)", dest_var="pred_mu", cable_kernel=dcable_cfg)
b_e = b.wire_to(e, src_var="phi(z)", dest_var="pred_mu", cable_kernel=dcable_cfg)
c_e = c.wire_to(e, src_var="phi(z)", dest_var="pred_targ", cable_kernel=dcable_cfg)
d_e = d.wire_to(e, src_var="phi(z)", dest_var="pred_targ", cable_kernel=dcable_cfg)

# wire error node e back to nodes a, b, c, and d to provide feedback to their states
e.wire_to(a, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(a_e,"symm_tied"))
e.wire_to(b, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(b_e,"symm_tied"))
e.wire_to(c, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(c_e,"symm_tied"))
e.wire_to(d, src_var="phi(z)", dest_var="dz_bu", mirror_path_kernel=(d_e,"symm_tied"))

# set up local Hebbian updates for a_e, b_e, c_e, and d_e
a_e.set_update_rule(preact=(a,"phi(z)"), postact=(e,"phi(z)"))
b_e.set_update_rule(preact=(a,"phi(z)"), postact=(e,"phi(z)"))
c_e.set_update_rule(preact=(a,"phi(z)"), postact=(e,"phi(z)"))
d_e.set_update_rule(preact=(a,"phi(z)"), postact=(e,"phi(z)"))
```

where we see that nodes `a` and `b` work together to deposit prediction signals
into `pred_mu` while nodes `c` and `d` jointly deposit signals into `pred_targ`
to create a target signal. Notice that we have wired `e` back to all four nodes
using a special flag/argument in the `wire_to()` routine, i.e., `mirror_path_kernel`,
which simply takes in a 2-tuple with the first element being the physical cable
object we want to reuse and the second is a string flag telling ngc-learn how
to re-use the cable (in this case, `symm_tied` means we use the transpose of the
underlying weight matrix contained inside the chosen dense cable).

Before we turn our attention to simulating the interactions/processing of the
above nodes and cables, there is one more specialized node worthy of mention --
the forward node or `FNode` (see [FNode](ngclearn.engine.nodes.fnode)).
This node is simple -- it only contains three
compartments: `dz`, `z`, and `phi(z)`. An `FNode` operates much like an `SNode`
except that it fundamentally is "stateless" -- external nodes deposit signals
into `dz` (where multiple deposits are vector summed) and then this value is
directly and automatically placed inside of `z` after which an encoded activation
function is applied to compute `phi(z)`. Note that an `SNode` can be modified
to also behave like an `FNode` by setting its argument `.zeta` equal to `0` and
setting `beta` to `1`. However, the `FNode` is a convenience node and is often
used to build an ancestral projection graph, of which we will describe later.

### Simulating Connected Nodes as Systems with the NGCGraph

Now that we have a basic grasp as to how nodes and cables can be instantiated
and connected to build neural circuits, let us examine the final key step required to
build an NGC system -- the simulation object `NGCGraph`
(see [NGCGraph](ngclearn.engine.ngc_graph)).

An `NGCGraph` is a general structure that will take in nodes that have been wired
together with cables and simulate their evolution/processing over time. This structure
crucially allows us to specify the execution sequence of nodes (or order of operations)
within a discrete step of simulation time.
It also provides several basic utility functions to facilitate analysis of the internal
nodes. In this demo, we will focus on the core primary routines one will need
to conduct most simulations, i.e., `set_cycle()`, `settle()`, `apply_constraints()`,
`calc_updates()`, `clear()`, and `extract()`.
Let us take the five node circuit we built above and place them in a system simulation:

```python
model = NGCGraph(K=5)
model.proj_update_mag = -1.0 # bound the calculated synaptic updates (<= 0 turns this off)
model.proj_weight_mag = 1.0 # constrain the Euclidean norm of the rows of each synaptic matrix
model.set_cycle(nodes=[a,b,c,d]) # execute nodes a through d (in order left to right)
model.set_cycle(nodes=[e]) # execute node e
model.apply_constraints() # immediately applies any constraints to synapses after initialization
```

where the above six lines of code create a full NGC system using the nodes and cables
we set earlier. The `set_cycle()` function takes in a list/array of nodes and
tells the underlying `NGCGraph` system to execute them (in the order of their appearance
within the list) first at each step in time. Making multiple subsequent calls
to `set_cycle()` will add in addition execution cycles to an NGC system's step.
Note that one simulation step of an `NGCGraph` consists of multiple cycles, executed
in the order of their calls when the simulation object was initialized. For example,
one step of our "model" object above would first execute the internal `.step()`
functions of `a`, `b`, `c`, then `d` in the first cycle and then execute the
`.step()` of `e` in the second cycle. Also observe that in our `NGCGraph` constructor,
we have told ngc-learn that simulations are only ever to be `K=5` steps long.
Finally note that, when you set execution cycles for an `NGCGraph`, ngc-learn
will examine the cables you wired between nodes and extract any learnable
synaptic weight matrices into a parameter object `.theta`.

With the above instantiation, we are now done building the NGC system and can
begin using it to process and adapt to sensory data. To make our five-node circuit
process a single data pattern, we would then write the following:

```python
x = tf.ones([1,10])
readouts = model.settle(
                clamped_vars=[("c","z",x)],
                readout_vars=[("a","phi(z)"),("b","phi(z)"),("d","phi(z)"),("e","phi(z)")]
            )
print("The value of {} w/in Node {} is {}".format(readouts[0][0], readouts[0][1], readouts[0][2].numpy()))
# update synaptic parameters given current model internal state
delta = model.calc_updates()
opt.apply_gradients(zip(delta, model.theta)) # apply some TF optimizer/adaptive learning rate here
model.apply_constraints()
model.clear() # reset the underlying node states back to resting values
```

where we have crafted a trivial example of processing a vector of ones (`x`),
clamping this value to node `c`'s compartment `z` (note that clamping means we
fix the node's compartment to a specific value and never let it evolves throughout
simulation), and then read out the value of the `phi(v)` compartment of nodes
`a`, `b`, `c`, and `e`. The `readout_vars` argument to `settle()` allows us to
tell an `NGCGraph` which nodes and which compartments we want to observe after it
run its simulated settling process over `K=5` steps. We saved the output of `settle()`
into the `readouts` variable which is a list of triplets of the form `[(node_name, comp_name, value),...]`
and in the example above we are deciding to print out the first node's value (in its set `phi(z)` compartment).
After ngc-learn executes its settling process, we can then tell it to update
all learnable synaptic weights (only for those cables that configured to use a
Hebbian update with `set_update_rule()`) via the `calc_updates()`, which itself
returns a list of the synaptic weight adjustments, in the order of the synaptic
matrices the `NGCGraph` object placed inside of `.theta`.

Desirably, after you have obtained `delta` from `calc_updates()`, you can then use
it with a standard Tensorflow 2 adaptive learning rate such as stochastic gradient
descent or Adam. An important point to understand is that an NGC system attempts
to maximize its total discrepancy, which is a negative quantity that it would like
to be at zero (meaning all local losses within it have reached an equilibrium at zero) --
this is akin to optimizing the approximate free energy of the system. Internally,
an `NGCGraph` will multiply the Hebbian updates by a negative coefficient to allow
the user to directly use a minimizer from a library such as Tensorflow.

After updating synaptic matrices using a Tensorflow optimizer, one then
calls `apply_constraints()` to ensure any weight matrix constraints are applied after
updating and finally ends with a call to `clear()`, which resets the values of all
nodes in the `NGCGraph` back to zero (or a resting state). (Note that if you do
not want the NGC system to reset its internal nodes back to resting zero states, then
simply do not call `clear()` until this effect is desired -- for example, on a
long temporal data stream such as a video feed, you might not want to reset the
NGC system back to resting until the video clip terminates).

## Learning a Data Generating Process: A Streaming NGC Model

Now that we familiarized ourselves with the basic mechanics of nodes and cables
and how they fit within a simulation graph, let's apply our knowledge and build
a nonlinear NGC generative model that learns to mimic a streaming data generating
process.

In ngc-learn, within the `generator` module, we have provided a few initial data
generators to facilitate prototyping and simulation studies. In this demonstration,
we will take a look at the `MoG` (mixture of Gaussians) static data generating process.













end
