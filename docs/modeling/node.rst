Node
====

A Node represents one of the fundamental building blocks of an op-graph system.
These particular objects are meant to perform, per simulated time step, a
calculation of output activity values given an internal arrangement of
compartments (or sources where signals from other Node(s) are to be deposited).

.. _node-model:

Node Model
--------------
The ``Node`` class serves as a root class for the node building block objects of
an op-graph. This is a core modeling component of general biomimetic computational
systems. Node sub-classes within ngc-learn inherit from this base class.

.. autoclass:: ngclearn.engine.nodes.node.Node
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
  .. automethod:: add_bundle_rule
    :noindex:
  .. automethod:: gather
    :noindex:
  .. automethod:: pre_gather
    :noindex:
  .. automethod:: dump
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:
