.. ngc-learn documentation master file, created by
   sphinx-quickstart on Wed Apr 20 02:52:17 2022.
   Note - This file needs to at least contain a root `toctree` directive.

Tutorial Contents
=================

Lessons/tutorials go through the very basics of constructing a dynamical system in
ngc-learn, core elements and tools of neurocognitive modeling using ngc-learn's 
in-built components and simulation tools, and finally providing foundational insights 
into how ngc-learn and its backend, ngc-sim-lib, work (particularly with respect 
to configuration). 

.. toctree::
  :maxdepth: 1
  :caption: I. Modeling Basics

  model_basics/configuration
  model_basics/json_modules
  model_basics/model_building
  model_basics/evolving_synapses

.. toctree::
  :maxdepth: 1
  :caption: II. Neurocognitive Modeling Lessons

  neurocog/index

.. toctree::
  :maxdepth: 1
  :caption: II. NGC-Learn/Sim-Lib Foundations

  foundations
  foundations/contexts
  foundations/commands
  foundations/operations

.. toctree::
  :maxdepth: 1
  :caption: III. NGC-Lava: Support for Loihi 2 Transfer

  lava/introduction
  lava/setup
  lava/lava_components
  lava/lava_context
