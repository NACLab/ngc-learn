# Tutorials

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

### Setting Up a Simple Model
The tutorials for ngc-learn are organized in the following manner:
1. <b>[Setting Up a JSON Experimental Configuration](../tutorials/model_basics/json_modules.md)</b>:
   In this lesson, you will learn how JSON configuration is used to rapidly
   allow you to import parts of ngc-learn relevant to the model you want to build.
2. <b>[Setting Up a Controller](../tutorials/model_basics/model_building.md)</b>: This
   lesson walk you through a basic setup of a controller with a few components.
3. <b>[Evolving Synaptic Parameter Values](../tutorials/model_basics/evolving_synapses.md)</b>: In this
   lesson, you will take your controller and configure its single synapse to
   evolve iteratively via a 2-factor Hebbian rule.
<!--6. <b>XXX</b>:-->

### Foundational Elements of ngc-learn and ngcsimlib
1. <b>[Understanding Commands](../tutorials/foundations/commands.md)</b>: This lesson will
   walk you through the basics of a command -- an essential part of building a
   simulation controller in ngc-learn and ngcsimlib -- and offer some useful
   points for designing new ones.
2. <b>[Bundle Rules](../tutorials/foundations/bundle_rules.md)</b>: Here, the basics
   of bundle rules, a commonly use mechanism for crafting complex biophysical
   systems, will be presented.
