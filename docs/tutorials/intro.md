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
and run simulations of computational biophysical models.

## Organization of Tutorials

The tutorials for ngc-learn are organized in the following manner:
1. <b>[Setting Up a JSON Experimental Configuration](../tutorials/lesson1.md)</b>:
   In this lesson, you will learn how to set up a JSON configuration to rapidly
   allow you to import parts of ngc-learn relevant to the model you want to build.
2. <b>[Instantiating a Component](../tutorials/lesson2.md)</b>:
3. <b>[Setting Up a Controller](../tutorials/lesson3.md)</b>:
4. <b>[Evolving Synaptic Parameter Values](../tutorials/lesson4.md)</b>: In this
   lesson, you will take your controller and configure its single synapse to
   evolve iteratively via a 2-factor Hebbian rule.
5. <b>XXX</b>: 
