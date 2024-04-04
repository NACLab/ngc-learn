# The Nodes-and-Cables System: The Component Building Block

The "component" is a core building block of the simulation construct that
ngc-learn and ngclib refer to as the controller (in effect, a controller, a
set of components embedded within it, and commands that drive the controller's
simulation process effectively make up ngc-learn's "nodes-and-cables system").
A controller, in short, maintains and does the bookkeeping for a set of
related components that compose the underlying simulation graph
(or operator graph) that represents a biomimetic
system in ngc-learn. Ultimately, when you create and connect components together,
you do so by placing them into a controller which will perform the calculations
in a pre-specified order -- this is what ngc-learn and ngclib use under the hood
to integrate the differential equations or recurrence relations that typically
describe a computational neuronal model. The final result: you can think of a
biomimetic model as a system of components that will be simulated across
time and all you need to do is tell the controller what components you want and
how they interact with one another.

## Biophysical Components

Concretely, in ngc-learn, you will deal with (or create) biophysical elements
that subclass the component class in ngclib. ngc-learn offers, at its base, a
set of fundamental biophysical components from a variety of categories -- these
include:
1. <b>[Neurons](../modeling/neurons.md)</b>: the neuronal cell is a foundational component allowing you to build
   the dynamics of interest, such as those related to leaky integrators that
   emit discrete action potentials or recurrent rate-coded neural states that
   follow a gradient flow (such as that of a free energy functional).
2. <b>[Synapses](../modeling/synapses.md)</b>: the synaptic cable is a key component that allows operators,
   such as neuronal cells, to communicate signals to one another. Synaptic cables
   facilitate the <i>message passing</i> between the cells of your system and
   generally are differentiated the way they project/transform signals and how
   they evolve with time (i.e., they specify their form of plasticity).
3. <b>[Input Encoders](../modeling/input_encoders.md)</b>: the input encoder is particularly useful when the
   simulated biomimetic system needs to encode its sensory inputs in a particular
   manner and this processing should be considered to a part of the model itself.
   A good example is a spiking neural network, where one typically wants to
   encode its sensory input, a normally static pixel image, as a series of action
   potentials; this can be done in ngc-learn using, for instance, a `PoissonCell`
   component, which produces a Poisson spike train under a certain desired
   maximum frequency.
4. <b>[Other Operators](../modeling/other_ops.md)</b>: these include mathematical transforms that may or
   may not have biological analogues. Some commonly-used ones include variable
   traces and kernels.
