# Synapses

The synapse is a key building blocks for connecting/wiring together the various
component cells that one would use for characterizing a biomimetic neural system.
These particular objects are meant to perform, per simulated time step, a
specific type of transformation utilizing their underlying synaptic parameters.
Most times, a synaptic cable will be represented by a set of matrices that
used to conduct a projection of an input signal (a value presented to its
pre-synaptic/input compartment) resulting in an output signal (a value that
appears within its post-synaptic compartment). Notably, a synapse component is
typically associated with a local plasticity rule, e.g., a Hebbian-type
update, that either is either triggered online (at some or all simulation time
steps) or by integrating a differential equation, e.g., via eligibility traces.

## Multi-Factor Learning Synapse Types

### (Two-Factor) Hebbian Synapse

```{eval-rst}
.. autoclass:: ngclearn.components.HebbianSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

## Spike-Timing-Dependent Plasticity (STDP) Synapse Types

### Trace-based STDP

```{eval-rst}
.. autoclass:: ngclearn.components.TraceSTDPSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Exponential STDP

```{eval-rst}
.. autoclass:: ngclearn.components.ExpSTDPSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```
