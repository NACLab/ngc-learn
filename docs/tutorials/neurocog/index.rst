.. ngc-learn documentation master file, created by
   sphinx-quickstart on Wed Apr 20 02:52:17 2022.
   Note - This file needs to at least contain a root `toctree` directive.

Neurocognitive Modeling Lessons
===============================

A central motivation for using ngc-learn is to flexibly build computational
models of neuronal information processing, dynamics, and credit
assignment (as well as design one's own custom instantiations of their
mathematical formulations and ideas). In this set of tutorials, we will go
through the central basics of using ngc-learn's in-built biophysical components,
also called "cells" and "synapses", to craft and simulate adaptive neural systems
and biophysical computational models.

Usefully, ngc-learn starts with a collection of cells -- those that are partitioned
into those that are graded / real-valued (`ngclearn.components.neurons.graded`)
and those that spike (`ngclearn.components.neurons.spiking`). In addition,
ngc-learn supports another collection called synapses -- generally, those that
adapt (or "learn") with biological credit assignment building blocks
(such as those in `ngclearn.components.synapses.hebbian`) such as
spike-timing-dependent plasticity and multi-factor rules. With the in-built,
standard cells and synapses in these two
core collections, you can readily construct a wide variety of models, recovering
many classical ones previously proposed in computational neuroscience
and brain-inspired computing researach (many of these kinds of models are available
for external download in the `Model Museum <https://github.com/NACLab/ngc-museum>`_).

While the reader is free to jump into any one self-contained tutorial in any
order based on their needs, we organize, within each topic, the lessons starting
from more basic, foundational modeling modules and library tools and sequentially
work towards more advanced concepts.

.. toctree::
  :maxdepth: 1
  :caption: Sensory Input Encoding / Transformation

  input_cells
  traces

.. toctree::
  :maxdepth: 1
  :caption: Spiking Neuronal Cells

  simple_leaky_integrator
  lif
  fitzhugh_nagumo_cell
  izhikevich_cell
  adex_cell
  hodgkin_huxley_cell

.. toctree::
  :maxdepth: 1
  :caption: Graded Neuronal Cells

  rate_cell
  error_cell

.. toctree::
  :maxdepth: 1
  :caption: Synapses and Forms of Plasticity

  dynamic_synapses
  hebbian
  stdp
  mod_stdp
  short_term_plasticity

.. toctree::
  :maxdepth: 1
  :caption: Model and Analysis Tools

  plotting
  metrics
  integration
  density_modeling
