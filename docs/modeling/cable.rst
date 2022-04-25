Cable
=====

A Cable represents one of the fundamental building blocks of an NGC system. These
particular objects are meant to serve as the connectors between Node(s), passing
along or transforming signals from the source point (a compartment, or receiving
area, within a particular node) to a destination point (another compartment in a
different node) and transforming such signals through synaptic parameters.

.. _cable-model:

Cable Model
--------------
The ``Cable`` class serves as a root class for the wire building block objects of an
NGC system/graph. This is a core modeling component of general NGC computational
systems. Cable sub-classes within ngc-learn inherit from this base class.

.. autoclass:: ngclearn.engine.cables.cable.Cable
  :noindex:

  .. automethod:: propagate
    :noindex:
  .. automethod:: set_update_rule
    :noindex:
  .. automethod:: calc_update
    :noindex:


.. _dcable-model:

DCable Model
--------------
The ``DCable`` class extends from the base ``Cable`` class and represents a
dense transform of signals from one nodal point to another. Signals that travel
across it through a set of synaptic parameters (and potentially a base-rate/bias
shift parameter).
In conjunction with the corresponding ``SCable`` class,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.cables.dcable.DCable
  :noindex:

  .. automethod:: propagate
    :noindex:
  .. automethod:: calc_update
    :noindex:


.. _scable-model:

SCable Model
--------------
The ``SCable`` class extends from the base ``Cable`` class and represents a
simple carry-over of signals from one nodal point to another. Signals that travel
across it can either be carried directly (an identity transform) or multiplied
by a scalar amplification coefficient.
In conjunction with the corresponding ``DCable`` class,
this serves as the core modeling component of a higher-level ``NGCGraph`` class
used in simulation.

.. autoclass:: ngclearn.engine.cables.scable.SCable
  :noindex:

  .. automethod:: propagate
    :noindex:
