# Introduction

ngc-learn is a general-purpose library for modeling biomimetic/neuro-mimetic
complex systems. While the library is designed to provide flexibility on the
experimenter/designer side -- allowing one to design their own dynamics and
evolutionary processes -- at its foundation are a few standard components, the
basic modeling nodes for simulating some common biophysical systems computationally,
that useful to know in getting started and quickly building some classical/historical
models. If you are interested in knowing some of the neurophysiological theory
behind ngc-learn's design philosophy, [this section](../tutorials/theory) might
be of interest.

Specifically, to make best use of ngc-learn, it is important to get the
hang of its "nodes-and-cables system" (as it was historically referred to) in
order to build simulation objects. This set of tutorials will walk through,
step-by-step, the key aspects of the library you need to know so you can build
and run simulations of computational biophysical models. In addition, we
provide walkthroughs of some of the central mechanisms underlying
<a href="https://github.com/NACLab/ngc-sim-lib">ngcsimlib</a>, the simulation
dependency library that drives ngc-learn; these are particularly useful for not
only understanding why and how things are done by ngc-learn's simulation
backend but also for those who want to design new, custom extensions of ngc-learn
either for their own research or to contribute to the development of the main library.

## Organization of Tutorials

### Setting Up and Modeling Basics
These [tutorials](../tutorials/modeling_basics.md) go through the key steps
and ideas of constructing a dynamical system in ngc-learn, including setting up a
[JSON configuration](../tutorials/model_basics/json_modules.md),
constructing a [basic model](../tutorials/model_basics/model_building.md),
and setting up a basic [form of plasticity](../tutorials/model_basics/evolving_synapses.md).

### Foundational Elements of ngc-learn and ngcsimlib
These [walkthroughs](../tutorials/foundations.md) go through the foundational
aspects of ngc-learn's simulation. Some current elements that walkthroughs are
provided for include:
[commands](../tutorials/foundations/commands.md) and
[bundle_rules](../tutorials/foundations/bundle_rules.md).
