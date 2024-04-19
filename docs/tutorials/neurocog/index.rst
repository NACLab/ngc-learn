.. ngc-learn documentation master file, created by
   sphinx-quickstart on Wed Apr 20 02:52:17 2022.
   Note - This file needs to at least contain a root `toctree` directive.

Neurocognitive Modeling Lessons
===============================

A central motivation for using ngc-learn is to flexibly build computational,
models of neuronal information processing, dynamics, and credit
assignment (as well as design one's own custom instantiations of their
mathematical formulations and ideas). In this set of tutorials, we will go
through the central basics of using ngc-learn's in-built biophysical components,
also called "cells", to craft and simulate adaptive neural systems.

Usefully, ngc-learn starts with a central partitioning of cells -- those that are
graded, real-valued (`ngclearn.components.neurons.graded`) and those that spike
(`ngclearn.components.neurons.spiking`). With the in-built, standard cells in
these two core partitions, you can readily construct a vast plethora of models,
recovering many classical models previously proposed in research in computational
neuroscience and brain-inspired computing (many of these models are available
for external download in the Model Museum, i.e., `model_museum`).

While the reader is free to jump into any one self-contained tutorial in any
order based on their needs, we organize, within each topic, the lessons starting
from more basic, foundational modeling modules and library tools and sequentially
work towards more advanced concepts.

.. toctree::
  :maxdepth: 1
  :caption: Sensory Input Encoding / Transformation

  input_cells

.. toctree::
  :maxdepth: 1
  :caption: Spiking Neuronal Cells

  simple_leaky_integrator
  fitzhugh_nagumo_cell
  izhikevich_cell

.. toctree::
  :maxdepth: 1
  :caption: Graded Neuronal Cells

  rate_cell
  error_cell

.. toctree::
  :maxdepth: 1
  :caption: Forms of Plasticity

  hebbian_plasticity
  stdp_plasticity

.. toctree::
  :maxdepth: 1
  :caption: Model Analysis Tools

  plotting
  metrics
