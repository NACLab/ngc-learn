# Introduction

NGC-Learn is a general-purpose library for modeling complex dynamical systems, particularly those that are useful for computational neuroscience, neuroscience-motivated artificial intelligence (NeuroAI), and brain-inspired computing. 
<!-- biomimetic/neuro-mimetic complex systems. --> 
While the library is designed to provide flexibility on the experimenter/designer side -- allowing one to develop their own dynamics and evolutionary processes -- at its foundation are a few standard components. These are basic modeling nodes for simulating some common biophysical systems computationally, which are useful to know when getting started and for quickly building some classical/historical models. If you are interested in knowing some of the neurophysiological theory behind NGC-Learn's design philosophy, [this section](../tutorials/theory) might be of interest.

Specifically, to make best use of NGC-Learn, it is important to get the hang of its "nodes-and-cables system" (the historical name for its backend engine) in order to build simulation objects. This set of tutorials will walk you through, step-by-step, the key aspects of the library that you will need to know so that you can build
and run simulations of computational biophysical models. In addition, we provide walkthroughs of some of the central mechanisms underlying <a href="https://github.com/NACLab/ngc-sim-lib">NGC-Sim-Lib</a>, the simulation dependency library that drives NGC-Learn; these lessons are particularly useful for not only understanding why and how things are done by NGC-Learn's simulation backend engine but also for those who want to design new, custom extensions of NGC-Learn either for their own research or to help contribute to the development of the main library.

## Organization of Tutorials

The core tutorials and lessons for using NGC-Learn can be found [here, in the tutorial table of contents](../tutorials/index.rst) which essentially go through: the basic configuration and use of NGC-Learn and NGC-Sim-Lib to construct simulations of dynamical systems, the essentials of neurocognitive modeling (such as building and analyzing models of neuronal dynamics and synaptic plasticity), as well as the coverage of some key foundational ideas/tools worth knowing about NGC-Learn (and its backend, NGC-Sim-Lib) particularly to facilitate easier debugging, experimental configuration, and advanced modeling tools. <!-- like `bundle rules`. -->

<!--
### Setting Up and Modeling Basics
These [tutorials](../tutorials/index.rst) go through the key steps
and ideas of constructing a dynamical system in ngc-learn, including setting up a
[JSON configuration](../tutorials/model_basics/json_modules.md),
constructing a [basic model](../tutorials/model_basics/model_building.md),
and setting up a basic [form of plasticity](../tutorials/model_basics/evolving_synapses.md).

### Neurocognitve Modeling Essentials
These [lessons](../tutorials/index.rst) go through how ngc-learn is used to 
build basic biophyiscal models and conduct simulation with neurocognitive 
systems.

### Foundational Elements of ngc-learn and ngcsimlib
These [walkthroughs](../tutorials/foundations.md) go through the foundational
aspects of ngc-learn's simulation. Some current elements that walkthroughs are
provided for include:
[commands](../tutorials/foundations/commands.md) and
[bundle_rules](../tutorials/foundations/bundle_rules.md).
-->
