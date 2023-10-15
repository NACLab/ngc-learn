OpGraph
=======

An OpGraph, or operator-graph, represents one of the core structural components
of a biomimetic system. This particular object is what Node(s) and cable(s) are
ultimately embedded into in order to simulate a full system process (key functions
include the primary global step and evolve routines).
Furthermore, the OpGraph contains several tool functions to facilitate
analysis of the system evolved over time.

.. _op-graph:

Op Graph
--------------
The ``OpGraph`` class serves as a core building block for forming a complete
biomimetic computational processing system.

.. autoclass:: ngclearn.engine.opgraph.OpGraph
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: set_to_rest
    :noindex:
  .. automethod:: probe
    :noindex:
  .. automethod:: clamp
    :noindex:
  .. automethod:: make_tracker
    :noindex:
  .. automethod:: track
    :noindex:
  .. automethod:: add_nodes
    :noindex:
  .. automethod:: add_cables
    :noindex:
  .. automethod:: add_cycle
    :noindex:
  .. automethod:: dump_graph
    :noindex:
  .. automethod:: load
    :noindex:
