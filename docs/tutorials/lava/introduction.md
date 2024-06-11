# Blending ngc-learn and lava-nc

The subpackage of ngclearn known as ngc-lava is an interfacing layer between 
ngclearn's components and contexts and lava-nc's models and processes. In this 
package, there is the introduction of the `LavaContext`, a subclass of the default
ngclearn `Context`. This context has all the same functionality as the base 
ngclearn context but adds the ability to convert lava compatible components into 
their Lava process and model automatically and on-the-fly. This allows for the
development and testing of models inside ngclearn prior to their deployment onto 
a Loihi neuromorphic chip without needing to translate between the two models 
written across the two different Python libraries.

## Some Cautionary Notes

- For the best experience in training models in ngclearn, Python version `>=3.10` 
  should be used. However, much of lava is written to be used in Python `3.8` and, 
  because of this, there are some flags and functionality that cannot be used in Lava
  components directly. It is for this reason that ngc-learn has several 
  in-built "lava components", i.e., those in `ngclearn.components.lava` that 
  are meant to directly interact with ngc-lava; other components (such as those
  (`ngclearn.components.neurons` or `ngclearn.components.synapses`) are not likely 
  to work and, when writing your own custom ngc-lava components, we recommend 
  that you use those in the `ngclearn.components.lava` subpackage as starting 
  points to see what design patterns will work with Lava.
- As of right now, all of ngc-lava is built using the Loihi2 configuration and  
  Loihi1 is not actively supported. Loihi1 might still work but nothing is 
  guaranteed nor has been tested by the ngc-learn dev team.

## Table of Contents
1. <b>[Setting up ngc-lava](setup.md)</b>: A brief overview of how to set up 
   ngc-lava
2. <b>[Lava components](lava_components.md)</b>: An overview of lava components in ngclearn and 
   how to make custom ones
3. <b>[Lava Context](lava_context.md)</b>: An overview of the Lava context and building 
   models for Lava
4. <b>[On-Chip Hebbian Learning](hebbian_learning.md): A walkthrough for getting a simple 
   hebbian learning model setup
