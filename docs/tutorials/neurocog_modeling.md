# Neurocognitive Modeling

A central motivation for using ngc-learn is to flexibly build computational,
models of neuronal information processing, dynamics, and credit
assignment (as well as design one's own custom instantiations of their
mathematical formulations and ideas). In this set of tutorials, we will go
through the central basics of using ngc-learn's in-built biophysical components,
also called "cells", to craft and simulate adaptive neural systems.

Usefully, ngc-learn starts with a central partitioning of cells -- those that are
graded, real-valued (`ngclearn.components.neurons.graded`) and those that spike
(`ngclearn.components.neurons.spiking`). With the in-built, standard cells in
these two core partitions, you can readily construct a vast plethora of models,
recovering many classical models previously proposed in research in computational
neuroscience and brain-inspired computing (many of these models are available
for external download in the Model Museum, i.e., `model_museum`).

While the reader is free to jump into any one self-contained tutorial in any
order based on their needs, we will start with cell components, examining
spiking neuronal cells first and then graded ones. After that, we will move on
to synaptic cables, starting with those that adapt via simple two-factor Hebbian
learning and then moving on those that do so with spike-timing-dependent
plasticity (STDP).
Finally, we will walk through several historical neuronal models that
are built with ngc-learn, ranging from predictive processing to sparse coding to
spiking systems to harmoniums.

Lessons are specifically organized in the following manner.

Input encoding cells are examined here:
1. <b>A</b>: Bernoulli and Poisson Spike Trains
2. <b>B</b>: Latency-coded Spike Train

Spiking cells are studied here:
1. <b>A</b>: [Simple Leaky Integrator](neurocog/simple_leaky_integrator.md)
2. <b>B</b>: A More Intricate Leaky Integrator-and-Fire
3. <b>C</b>: The [FitzHugh–Nagumo Neuronal Model](neurocog/fitzhugh_nagumo_cell.md)
4. <b>D</b>: The Izhikevich Neuronal Model

Graded cells are studied here:
1. <b>A</b>: [The Rate Cell Model](neurocog/rate_cell.md)
2. <b>B</b>: [Error Cell Models](neurocog/error_cell.md)

Synaptic cables are studied here:
1. <b>A</b>: The Hebbian Synapse Model
2. <b>B</b>: Trace-based Spike-Timing-Dependent Plasticity (STDP)

Neuronal models that will be examined (using `ngc-museum`) here:
1. Sparse Coding
2. Deep Iterative Thresholding (and Amortized Inference)
3. Predictive Coding (Discriminative)
4. Predictive Coding (Generative)
5. Broadcast Feedback Alignment and Spiking Neurons
6. Harmoniums (Restricted Boltzmann Machines)
