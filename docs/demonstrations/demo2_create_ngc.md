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
For example, a five-node system could be as follows:

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
pattern as in the code snippet above.

To learn or adjust the synaptic cables connecting the nodes in the five-node
system we created above, we need to configure the cables themselves to use
a local Hebbian-like update. For example, if we want the cable `a_c` to evolve
over time, we notify it that it needs to update according to:

```python
a_c.set_update_rule(preact=(a,"phi(z)"), postact=(c,"phi(z)"))
```

where the above sets a Hebbian update that will compute an update/adjustment
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




...

## Learning a Data Generating Process: A Streaming NGC Model

...














end
