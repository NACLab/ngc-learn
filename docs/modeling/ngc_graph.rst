NGCGraph
========

An NGCGraph represents one of the core structural components of an NGC system.
This particular object is what Node(s) and Cable(s) are ultimately
embedded/integrated into in order to simulate a full NGC process (key functions
include the primary settling process and synaptic update calculation routine).
Furthermore, the NGCGraph contains several tool functions to facilitate
analysis of the system evolved over time.

.. _ngc-graph:

NGC Graph
--------------
The ``NGCGraph`` class serves as a core building block for forming a complete
NGC computational processing system.

.. autoclass:: ngclearn.engine.ngc_graph.NGCGraph
  :noindex:

  .. automethod:: set_cycle
    :noindex:
  .. automethod:: compile
    :noindex:
  .. automethod:: clone_state
    :noindex:
  .. automethod:: set_to_state
    :noindex:
  .. automethod:: extract
    :noindex:
  .. automethod:: getNode
    :noindex:
  .. automethod:: clamp
    :noindex:
  .. automethod:: inject
    :noindex:
  .. automethod:: settle
    :noindex:
  .. automethod:: calc_updates
    :noindex:
  .. automethod:: apply_constraints
    :noindex:
  .. automethod:: clear
    :noindex:
