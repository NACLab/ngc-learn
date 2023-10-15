Op
==

An Op, a special sub-class of Node, is a representation of a mathematical
transform, which may or may not have a biophysical analogue/metaphor, that
is executed within an op-graph system. Mathematical operators should provide
operational functionality ranging from analysis to signal filtering.

.. _op-model:

Operator Model
---------------
The ``Op`` class serves as a base mathematical transform that sub-classes the
op-graph's base Node root class. Op sub-classes within ngc-learn inherit from
this base class.

.. autoclass:: ngclearn.engine.nodes.ops.op.Op
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


.. _vartrace-model:

Variable Trace Model
---------------------
The ``VarTrace`` sub-class extends from the base ``Op`` class to provide a
variable trace filter, with flags for switching between linear and exponential
decay as well as between continuously incremental and step-like behavior.

.. autoclass:: ngclearn.engine.nodes.ops.varTrace.VarTrace
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: get_default_out
    :noindex:
  .. automethod:: custom_dump
    :noindex:


.. _expkernel-model:

Exponential Kernel Model
-------------------------
The ``ExpKernel`` sub-class extends from the base ``Op`` class to provide a
spike response function based on an exponential kernel applied to a moving
window of continuous spike times.

.. autoclass:: ngclearn.engine.nodes.ops.expKernel.ExpKernel
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: set_to_rest
    :noindex:
  .. automethod:: get_default_out
    :noindex:
  .. automethod:: custom_dump
    :noindex:


.. sum-model:

Summation Model
-------------------------
The ``SumNode`` sub-class extends from the base ``Op`` class to provide a
generalized summation operation -- this node-op will literally element-wise
sum its input compartments, as dictated by its additive bundle function.

.. autoclass:: ngclearn.engine.nodes.ops.sumNode.SumNode
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: pre_gather
    :noindex:
  .. automethod:: custom_dump
    :noindex:


.. scale-model:

Scaling Model
-------------------------
The ``ScaleNode`` sub-class extends from the base ``Op`` class to provide a
simple scaling of its input compartment (i.e., multiplying it by a scalar value).

.. autoclass:: ngclearn.engine.nodes.ops.scaleNode.ScaleNode
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: custom_dump
    :noindex:
