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
Each A has different, multiple "compartments" (which we allow to be named, if desired),
which allows it to collect information from many different connected/related nodes
and then, within its integration routine (or `step()`), decide how to combine the
different signals in order to calculate its own activity (loosely corresponding to a
rate-coded firing rate -- we will learn how to model spike trains in a later
demonstration). As a result, many nodes and cables yield an NGC system where each
node is itself, in general, a stateful computation (even if we are processing static
data such as images).

## Building NGC Systems with Nodes and Cables
With the above aspect of ngc-learn's theoretical framing in mind, we can craft
connectivity patterns of our own by deciding the form that each node and cable
in our system will take. ngc-learn currently offers a few core nodes and cable types
(note ngc-learn is an evolving software framework, so more node/cable types are to come
in future releases, either through the NAC team or community contributions).
The core node type set currently includes `SNode`, `ENode`, and `FNode` (all inheriting
from the `Node` base class) while the current cable type set includes `DCable` and
`SCable` (all inherited from the `Cable` base class).

An `SNode` refers to a stateful node, which is one of the primary nodes you will
work with when crafting NGC systems. A stateful node contains inside a cluster
(or block) of neurons, the number of which is controlled through the `dim`
argument.















end
