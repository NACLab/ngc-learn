# Blending ngc-learn and lava-nc

The subpackage of ngclearn known as ngc-lava is an interfacing layer between ngclearn's components and contexts, and
lava-nc's models and processes. In this package there is the introduction of the `LavaContext` a subclass of the default
ngclearn context. This context has all the same functionality as the base ngclearn context but adds the ability to
convert lava compatible components into their lava process and model automatically and on the fly. This allows for the
development and testing of models inside ngclearn prior to their deployment onto a Loihi chip without needing to
translate the two models.

### Prior Warnings

- For the best experience training models in ngclearn python version `>=3.10` should be used, however much of lava is
  written to be used in python `3.8` because of this there are some flags and functionality that can not be used in lava
  components directly.
- As of right now all of ngc-lava is built using the Loihi2 configuration and the Loihi1 is not actively supported. It
  might still work but nothing is guaranteed or tested.


## Table of contents
1. <b>[Setting up ngc-lava](setup.md)</b>: A brief overview of how to set up ngc-lava
2. 


## Setup
To set up a project using ngc-lava first clone the repo from `https://github.com/NACLab/ngc-lava`