# Theory and Design Motivation

## Cable Theory and Neural Compartments
At its core, part of ngc-learn's internal design is inspired by (neural) 
<a href="http://www.scholarpedia.org/article/Neuronal_cable_theory">cable theory </a>, 
where neuronal units, which are arranged in complex connectivity structures, are viewed 
as performing dendritic calculations (of varying complexity). In essence, a particular 
neuron integrates information from different input signal sources (for example, 
signals produced by other neurons), in often highly nonlinear ways through a 
complex dendritic tree.

Although modeling a complete neuronal system through the lens of cable theory is 
complex and intricate in of itself, ngc-learn is built with this direction in 
mind. ngc-learn starts with with the idea that a neuron (or a cluster of them) 
can be viewed as a node or nodal component -- specifically a type of "cell" 
component (in ngc-learn, many of these are component classes that end with the 
suffix `Cell`) -- and each bundle of synapses that connects pairs of nodes can 
be viewed as a cable  -- specifically a "synapse" component  (these component 
classes usually end with the suffix `Synapse` or `SynapticCable`)-- that performs 
some sort of transformation of its pre-synaptic signal (also treated as another 
component in terms of abstract simulation) and often differentiated by its form 
of plasticity. See the [Neurons](../modeling/neurons) specification for the base available 
neuronal cells and the [Synapses](../modeling/synapses) specification for the base available 
synaptic cables. Note that these two types of nodal components can be combined 
with other types such as [Input Encoders](../modeling/input_encoders) and [Operations](../modeling/other_ops) to build 
gradually more complex dynamical biomimetic/neuro-mimetic systems.

Each neuronal cell component/node has multiple, different (named) "compartments", 
which are regions or slots within the node that other nodes can deposit 
information/signals into. These compartments allow a node to collect information 
from many different connected/related nodes and then decide how to combine these 
different signals in order calculate its own output activity (either in the form 
of a rate-coded firing rate or binary spikes) using the integration logic defined 
within its own specific `advanceState()` function. When a biomimetic system, 
composed of many of these nodes/components, is simulated over a period of time 
(processing some form of sensory input), its underlying simulation object 
(the `Controller`) calls the `advanceState()` routine of each constituent node, 
shifting that nodes internal time by one discrete step. The order in which the 
node `advanceState()` routines are called is governed by "run cycles", which are 
defined by the experimenter at the object initialization of the controller. For 
example, a user might want one set of nodes to first execute their internal step 
logic before another set is able to -- this could be done by specifying two 
distinct cycles in the order desired.

As a result, many nodes, and the synaptic cables that connect them, result in 
a simulated biomimetic system where each node is itself, in general, treated as 
a stateful computation even if we are processing inherently non-temporal data 
such as static images.
