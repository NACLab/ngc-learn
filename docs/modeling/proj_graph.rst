ProjectionGraph
===============

A ProjectionGraph represents one of the core structural components of an
NGC system.
To use a projection graph, Node(s) and Cable(s) must be embedded/integrated
into in order to simulate an ancestral projection/sampling process.
Note that ProjectionGraph is only useful if an NGCGraph has been created,
given that a projection graph is meant to offer non-trainable functionality,
particularly fast inference, to an NGC computational system.

.. _proj-graph:

Projection Graph
----------------
The ``ProjectionGraph`` class serves as a core building block for forming a complete
NGC computational processing system (particularly with respect to external
processes such as projection/sampling).

.. autoclass:: ngclearn.engine.proj_graph.ProjectionGraph
  :noindex:

  .. automethod:: set_cycle
    :noindex:
  .. automethod:: extract
    :noindex:
  .. automethod:: getNode
    :noindex:
  .. automethod:: project
    :noindex:
  .. automethod:: clear
    :noindex:
