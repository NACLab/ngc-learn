# Introduction

NGC-Learn is a general-purpose library for modeling complex dynamical systems, particularly those that are useful for 
computational neuroscience, neuroscience-motivated artificial intelligence (NeuroAI), and brain-inspired computing. 
<!-- biomimetic/neuro-mimetic complex systems. --> 
While the library is designed to provide flexibility on the experimenter/designer side -- allowing one to develop their 
own dynamics and evolutionary processes -- at its foundation are a few standard components. These are basic modeling 
nodes for simulating some common biophysical systems computationally, which are useful to know when getting started and 
for quickly building some classical/historical models. If you are interested in knowing some of the neurophysiological 
theory behind NGC-Learn's design philosophy, [this section](../tutorials/theory) might be of interest.

Specifically, to make best use of NGC-Learn, it is important to get the hang of its "nodes-and-cables system" (the 
historical name for its backend engine) in order to build simulation objects. This set of tutorials will walk you 
through, step-by-step, the key aspects of the library that you will need to know so that you can build
and run simulations of computational biophysical models. In addition, we provide walkthroughs of some of the central 
mechanisms underlying <a href="https://github.com/NACLab/ngc-sim-lib">NGC-Sim-Lib</a>, the simulation dependency 
library that drives NGC-Learn; these lessons are particularly useful for not only understanding why and how things are 
done by NGC-Learn's simulation backend engine but also for those who want to design new, custom extensions of NGC-Learn 
either for their own research or to help contribute to the development of the main library.

## Organization of Tutorials

The core tutorials and usage lessons for using NGC-Learn can be found [here, in the modeling basics table of contents](../tutorials/index.rst) which essentially go through: the basic configuration and use of NGC-Learn (and NGC-Sim-Lib) to 
construct simulations of basic dynamical systems. 
More advanced tutorials related to the essentials of neurocognitive modeling -- such as building and analyzing 
neuroscience models of neuronal dynamics and synaptic plasticity -- can be found [here, in the neurocognitive modeling 
table of contents](../tutorials/neurocog/index.rst).
