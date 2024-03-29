.. ngc-learn documentation master file, created by
   sphinx-quickstart on Wed Apr 20 02:52:17 2022.
   Note - This file needs to at least contain a root `toctree` directive.

Welcome to ngc-learn's documentation!
=====================================

**ngc-learn** is a Python library for building, simulating, and analyzing
biomimetic computational models and arbitrary
predictive processing/coding models based on the neural generative
coding (NGC) computational framework. This toolkit is built on top of
`JAX <https://github.com/google/jax>`_ and is distributed under the
3-Clause BSD license.

.. toctree::
   :maxdepth: 1
   :caption: Introduction:

   overview
   installation

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  tutorials/intro
  tutorials/theory
  tutorials/lesson1_modules
  tutorials/lesson2
  tutorials/lesson3
  tutorials/lesson4

.. toctree::
  :maxdepth: 2
  :caption: Modeling API

  modeling/components
  modeling/neurons
  modeling/synapses
  modeling/input_encoders
  modeling/other_ops

.. toctree::
  :maxdepth: 1
  :caption: Model Museum

  museum

.. toctree::
   :maxdepth: 6
   :caption: Source API

   source/modules

.. toctree::
  :maxdepth: 1
  :caption: Papers that use NGC-Learn

  ngclearn_papers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
