# Neuronal Cells

The neuron (or neuronal cell) represents one of the fundamental building blocks
of a biomimetic neural system. These particular objects are meant to perform,
per simulated time step, a calculation of output activity values given an
internal arrangement of compartments (or sources where signals from other
neuronal cell(s) are to be deposited). Typically, a neuron integrates an
(ordinary) differential equation, which depends on the type of neuronal
cell and dynamics under consideration.

## Graded, Real-valued Neurons

This family of neuronal cells adheres to dynamics or performs calculations
utilizing graded (real-valued/continuous) values; in other words, they do not
produce any discrete signals or action potential values.

### The Rate Cell

This cell evolves one set of dynamics over state `z` (a sort of real-valued
continuous membrane potential). The "electrical" inputs that drive it include
`j` (non-modulated signals) and `j_td` (modulated signals), which can be mapped
to bottom-up and top-down pressures (such as those produced by error neurons) if
one is building a strictly hierarchical neural model. Note that the "spikes" `zF`
emitted are real-valued for the rate-cell and are represented via the application
of a nonlinear activation function (default is the `identity`) configured by
the user.

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

This cell is (currently) a stateless neuron, i.e., it is not driven by an
underlying differential equation, emulating a "fixed-point" error or mismatch
calculation. Specifically, this cell is (currently) fixed to be a Gaussian
cell that assumes an identity covariance. Note that this neuronal cell has
several important compartments: in terms of input compartments, `target` is
for placing the desired target activity level while `mu` is for placing an
externally produced mean prediction value, while in terms of output
compartments, `dtarget` is the first derivative with respect to the target
(sometimes used to emulate a top-down pressure/expectation in predictive coding)
and `dmu` is the first derivative with respect to the mean parameter. (Note that
these compartment values depend on the distribution assumed by the error-cell.)

Variations of the fixed-point error cell depend on the local distribution assumed
over mismatch activities, e.g., Gaussian distribution yields a Gaussian error cell.

#### Gaussian Error Cell

```{eval-rst}
.. autoclass:: ngclearn.components.GaussianErrorCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

#### Laplacian Error Cell

```{eval-rst}
.. autoclass:: ngclearn.components.LaplacianErrorCell
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

### The Simplified LIF (sLIF) Cell

This cell, which is a simplified version of the leaky integrator (i.e., model
described later below), models dynamics over voltage `v` and threshold `thr`
(note that `j` is further treated as a point-wise current for simplicity).
Importantly, an optional fast form of lateral inhibition can be emulated with
this cell by setting the inhibitory resistance `inhibit_R > 0` -- this will mean
that the dynamics over `v` include a term that is equal to a negative hollow
matrix product with the spikes emitted at time `t-1` (yielding a recurrent
negative pressure on the membrane potential values at `t`).

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

### The Quadratic LIF (Leaky Integrator) Cell

This cell models dynamics over the voltage `v` and threshold shift `thrTheta`
(a homeostatic variable) (with optional dynamics for current `j`). Note that
`thr` is used as a baseline level for the membrane potential threshold while
`thrTheta`  is treated as a form of short-term plasticity (full threshold
is: `thr + thrTheta(t)`). The dynamics are driven by a critical voltage value
as well as a voltage scaling factor for membrane potential accumulation over time.

```{eval-rst}
.. autoclass:: ngclearn.components.QuadLIFCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Fitzhugh-Nagumo Cell

This cell models dynamics over voltage `v` and a recover variable `w` (which
governs the behavior of the action potential of a spiking neuronal cell). In
effect, the Fitzhugh-Nagumo model is a set of two coupled differential equations
that simplify the four differential equation Hodgkin-Huxley (squid axon) model. 
A voltage `v_thr` can be used to extract binary spike pulses.

```{eval-rst}
.. autoclass:: ngclearn.components.FitzhughNagumoCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```
