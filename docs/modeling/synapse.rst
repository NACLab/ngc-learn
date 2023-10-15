Synapse
=======

A Synapse, a special sub-class of Node, is a biophyscial, explicit representation
of a transform applied across a cable, which are implicit objects themselves.
A synapse may or may not implement a form of plasticity and thus warrant a call
to its evolve(.) sub-routine.

.. _synapse-model:

Synapse Model
--------------
The ``Synapse`` class serves as the biophyscial analogue of the op-graph's
implicit cable construct and is a special sub-class of the Node root class.
``Synapse`` sub-classes within ngc-learn inherit from this base class.
The Synapse construct serves as the core modeling component of a higher-level
``OpGraph`` class used in simulation of a biomimetic system.

.. autoclass:: ngclearn.engine.nodes.synapses.synapse.Synapse
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: set_to_rest
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


.. _projection-model:

Projection Synapse Model
-------------------------
The ``ProjectionSynapse`` class extends from the base ``Synapse`` class to
represent a dense transform of signals from one cell node's compartment to
that of another. There is NOT plasticity inherent to this synaptic transform.

.. autoclass:: ngclearn.engine.nodes.synapses.projectionSynapse.ProjectionSynapse
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:


.. _hebbian-model:

Hebbian Synapse Model
-----------------------
The ``HebbianSynapse`` class extends from the base ``Synapse`` class to
represent a dense transform of signals from one cell node's compartment to
that of another. The plasticity inherent to this synapse is based on a
multi-factor Hebbian process.

.. autoclass:: ngclearn.engine.nodes.synapses.hebbianSynapse.HebbianSynapse
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:


.. evstdp-model:

Event-based STDP Synapse Model
--------------------------------
The ``EvSTDPSynapse`` class extends from the base ``Synapse`` class to
represent a dense transform of signals from one cell node's compartment to
that of another. The plasticity inherent to this synapse is based on
post-synaptic event-driven STDP.

.. autoclass:: ngclearn.engine.nodes.synapses.evSTDPSynapse.EvSTDPSynapse
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:


.. _trstdp-model:

Traced-based STDP Synapse Model
---------------------------------
The ``TrSTDPSynapse`` class extends from the base ``Synapse`` class to
represent a dense transform of signals from one cell node's compartment to
that of another. The plasticity inherent to this synapse is based on
traced-based STDP (including both or only one of pre- and post-synaptic
updating schemes). Concretely implements power-law and exponential STDP
plasticity.

.. autoclass:: ngclearn.engine.nodes.synapses.trSTDPSynapse.TrSTDPSynapse
  :noindex:

  .. automethod:: step
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: custom_dump
    :noindex:
  .. automethod:: custom_load
    :noindex:
