.. ngc-learn documentation master file, created by
   sphinx-quickstart on Wed Apr 20 02:52:17 2022.
   Note - This file needs to at least contain a root `toctree` directive.

Tutorial Contents
=================

Lessons go through the very basics of constructing a dynamical system in
ngc-learn while foundations go through the fundamental aspects of ngc-learn's
simulation schema and design.

.. toctree::
  :maxdepth: 1
  :caption: Modeling Basics

  modeling_basics
  model_basics/configuration
  model_basics/json_modules
  model_basics/model_building
  model_basics/evolving_synapses

.. toctree::
  :maxdepth: 1
  :caption: Foundations

  foundations
  foundations/commands
  foundations/bundle_rules

.. toctree::
  :maxdepth: 1
  :caption: Neurocognitive Modeling

  neurocog_modeling
  neurocog/input_cells
  neurocog/simple_leaky_integrator
  neurocog/fitzhugh_nagumo_cell
  neurocog/izhikevich_cell
  neurocog/rate_cell
  neurocog/error_cell
