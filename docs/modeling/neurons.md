# Neuronal Cells

The neuron (or neuronal cell) represents one of the fundamental building blocks
of a biomimetic neural system. These particular objects are meant to perform,
per simulated time step, a calculation of output activity values given an
internal arrangement of compartments (or sources where signals from other
neuronal cell(s) are to be deposited). Typically, a neuron integrates an
(ordinary) differential equation, which depends on the type of neuronal
cell and dynamics under consideration.

## Rate-coded / Real-valued Neurons

### The Rate Cell

```{eval-rst}
.. autoclass:: ngclearn.components.RateCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Error Cell

```{eval-rst}
.. autoclass:: ngclearn.components.ErrorCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

## Spiking Neurons

These neuronal cells exhibit dynamics that involve emission of discrete action
potentials (or spikes). Typically, such neurons are modeled with multiple
compartments, including at least one for the electrical current `j`, the
membrane potential `v`, the voltage threshold `thr`, and action potential `s`.
Note that the interactions or dynamics underlying each component might itself
be complex and nonlinear, depending on the neuronal cell simulated (i.e., some
neurons might be running multiple differential equations under the hood).

### The LIF (Leaky Integrator) Cell

This cell models dynamics over the voltage `v` and threshold shift `thrTheta`
(a homeostatic variable) (with optional dynamics for current `j`). Note that
`thr` is used as a baseline level for the membrane potential threshold while
`thrTheta`  is treated as a form of short-term plasticity (full threshold
is: `thr + thrTheta(t)`).

```{eval-rst}
.. autoclass:: ngclearn.components.LIFCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Simplified LIF (sLIF) Cell

This cell, which is a simplified version of the leaky integrator, models dynamics
over voltage `v` and threshold `thr` (note that `j` is further treated as a
point-wise current for simplicity). Importantly, an optional fast form of
lateral inhibition can be emulated with this cell by setting the inhibitory
resistance `inhibit_R > 0` -- this will mean that the dynamics over `v` include
a term that is equal to a negative hollow matrix product with the spikes
emitted at time `t-1` (yielding a recurrent negative pressure on the membrane
potential values at `t`).

```{eval-rst}
.. autoclass:: ngclearn.components.SLIFCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```
