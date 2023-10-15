Cell
====

A Cell, a special sub-class of Node, is a representation of a biophysical
entity that will be simulated within an op-graph system. This is where
mathematical models of multi-compartmental neuronal cell units should be
implemented.

.. _cell-model:

Cell Model
--------------
The ``Cell`` class serves as the biophyscial analogue of the op-graph's base
Node root class. Cell sub-classes within ngc-learn inherit from this base class.

.. autoclass:: ngclearn.engine.nodes.cells.cell.Cell
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: set_to_rest
    :noindex:
  .. automethod:: clamp
    :noindex:
  .. automethod:: get_default_in
    :noindex:
  .. automethod:: get_default_out
    :noindex:
  .. automethod:: get
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:


.. _passcell-model:

Pass-through Cell Model
------------------------
The ``PassCell`` extends from the base ``Cell`` class and represents a
simple transmission of input to output (pass-through) or the identity function.

.. autoclass:: ngclearn.engine.nodes.cells.passCell.PassCell
  :noindex:

  .. automethod:: step
    :noindex:


.. _errorcell-model:

Error Cell Model
-----------------
The ``ErrCell`` extends from the base ``Cell`` class and represents a
(rate-coded) error node simplified to its fixed-point form.

.. autoclass:: ngclearn.engine.nodes.cells.errCell.ErrCell
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: get_default_in
    :noindex:
  .. automethod:: get_default_out
    :noindex:


.. _poisscell-model:

Poisson Cell Model
-------------------
The ``PoissCell`` extends from the base ``Cell`` class and represents an
approximate Poisson spike cell, using a Bernoulli trial spikes.

.. autoclass:: ngclearn.engine.nodes.cells.poissCell.PoissCell
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: get_default_out
    :noindex:


.. _latencycell-model:

Latency Cell Model
-------------------
The ``LatencyCell`` extends from the base ``Cell`` class and represents a
(nonlinear) latency encoding (spike) cell/function.

.. autoclass:: ngclearn.engine.nodes.cells.latencyCell.LatencyCell
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: get_default_out
    :noindex:


.. _lifcell-model:

LIF Cell Model
-----------------
The ``LIFCell`` extends from the base ``Cell`` class and represents a spiking
neuronal cell based on leaky integrate-and-fire (LIF) dynamics.

.. autoclass:: ngclearn.engine.nodes.cells.LIFCell.LIFCell
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: set_to_rest
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:
  .. automethod:: get_default_in
    :noindex:
  .. automethod:: get_default_out
    :noindex:


.. _quadlifcell-model:

Quadratic LIF Cell Model
-------------------------
The ``QuadLIFCell`` extends from the ``LIF`` class and represents a
spiking neuronal cell based on quadratic leaky integrate-and-fire (quad-LIF)
dynamics.

.. autoclass:: ngclearn.engine.nodes.cells.quadLIFCell.QuadLIFCell
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: get_default_in
    :noindex:
  .. automethod:: get_default_out
    :noindex:


.. _wtascell-model:

Winner-take-all Score Model
----------------------------
The ``WTASCell`` extends from the base ``Cell`` class and represents a
scoring-based spike response function that ensures only one neuron spikes
at any time step.

.. autoclass:: ngclearn.engine.nodes.cells.WTASCell.WTASCell
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: set_to_rest
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:
  .. automethod:: get_default_in
    :noindex:
  .. automethod:: get_default_out
    :noindex:
