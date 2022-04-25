Node
====

.. _node-model:

Node Model
--------------
The ``Node`` class serves as a root class and fundamental building block of an
NGC system/graph. This is a core modeling component of general NGC computational
systems. Node sub-classes within ngc-learn inherit from this base class.

.. autoclass:: ngclearn.engine.nodes.node.Node
  :noindex:

  .. automethod:: wire_to
    :noindex:
  .. automethod:: inject
    :noindex:
  .. automethod:: clamp
    :noindex:
  .. automethod:: step
    :noindex:
  .. automethod:: calc_update
    :noindex:
  .. automethod:: clear
    :noindex:
  .. automethod:: extract
    :noindex:
  .. automethod:: extract_params
    :noindex:
  .. automethod:: deep_store_state
    :noindex:


.. _snode-model:

SNode Model
-----------
The ``SNode`` class extends from the base ``Node`` class, and represents
a (rate-coded) state node that follows a certain set of settling dynamics.
In conjunction with the corresponding ``ENode`` and ``FNode`` classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.snode.SNode
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: clear
    :noindex:


.. _enode-model:

ENode Model
-----------
The ``ENode`` class extends from the base ``Node`` class, and represents
a (rate-coded) error node simplified to its fixed-point form.
In conjunction with the corresponding ``SNode`` and ``FNode`` classes,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.nodes.enode.ENode
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: calc_update
    :noindex:
  .. automethod:: compute_precision
    :noindex:
  .. automethod:: clear
    :noindex:
