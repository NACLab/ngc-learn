# Demo 2: Creating Custom NGC Systems

## Theoretical Inspiration/Grounding
In this demonstration, we will learn how to craft our own custom NGC system
using ngc-learn's fundamental building blocks -- nodes and cables. At its core,
part of ngc-learn's fundamental design is inspired by
<a href="http://www.scholarpedia.org/article/Neuronal_cable_theory">cable theory </a>,
where neurons, arranged in complex connectivity structures, are viewed as
performing dendritic calculations. In other words, a particular neuron integrates
information from different input signals (for example, those from other neurons), in
often highly nonlinear ways through a complex dendritic tree. While modeling
a neuronal system through the lens of cable theory is certainly complex and intricate
in of itself, ngc-learn is built in this direction, starting with the idea a neuron
(or a cluster of them) can be viewed as a node (`Node`) and each bundle of synapses
that connect nodes can be viewed as a cable (`Cable`). Each node has differnt, multiple
"compartments" (which we allow to be named), which allows it to collect information
from many different nodes and then, within its integration routine (or `step()`),
decide how to combine the different signals in order to calculate its own activity
(loosely corresponding to a rate-coded firing rate). Many nodes and cables yield
an NGC system where each node is itself, in general, a stateful computation (even
if we are processing static data such as images).

## Building NGC Systems with Nodes and Cables
With some of ngc-learn's theoretical framing in mind, we can craft connectivity
patterns of our own by simply deciding the form that nodes and cables in our system
will take. ngc-learn currently offers a few core nodes and cable types (noting that
ngc-learn is meant to be an evolving software framework, so many more are to come
in future releases either from the NAC team or through community contributions).
The core node type set currently includes `SNode`, `ENode`, and `FNode` (all inheriting
from the `Node` base class) while the current cable type set includes `DCable` and
`SCable` (all inherited from the `Cable` base class).
