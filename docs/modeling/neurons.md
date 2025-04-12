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
  .. automethod:: reset
    :noindex:
```

### The Error Cell

This cell is (currently) a stateless neuron, i.e., it is not driven by an
underlying differential equation, thus emulating a "fixed-point" error or mismatch
calculation. Variations of the fixed-point error cell depend on the local
distribution assumed over mismatch activities, e.g., Gaussian distribution
yields a Gaussian error cell, which will also change the form of their
internal compartments (typically a `target`, `mu`, `dtarget`, and `dmu`).

#### Gaussian Error Cell

This cell is (currently) fixed to be a Gaussian
cell that assumes an identity covariance. Note that this neuronal cell has
several important compartments: in terms of input compartments, `target` is
for placing the desired target activity level while `mu` is for placing an
externally produced mean prediction value, while in terms of output
compartments, `dtarget` is the first derivative with respect to the target
(sometimes used to emulate a top-down pressure/expectation in predictive coding)
and `dmu` is the first derivative with respect to the mean parameter.

```{eval-rst}
.. autoclass:: ngclearn.components.GaussianErrorCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

#### Laplacian Error Cell

This cell is (currently) fixed to be a Laplacian
cell that assumes an identity scale. Note that this neuronal cell has
several important compartments: in terms of input compartments, `target` is
for placing the desired target activity level while `mu` is for placing an
externally produced mean prediction value, while in terms of output
compartments, `dtarget` is the first derivative with respect to the target
(sometimes used to emulate a top-down pressure/expectation in predictive coding)
and `dmu` is the first derivative with respect to the mean parameter.

```{eval-rst}
.. autoclass:: ngclearn.components.LaplacianErrorCell
  :noindex:

  .. automethod:: advance_state
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
  .. automethod:: reset
    :noindex:
```

### The LIF (Leaky Integrate-and-Fire) Cell

This cell (the "leaky integrator") models dynamics over the voltage `v`
and threshold shift `thrTheta` (a homeostatic variable). Note that `thr`
is used as a baseline level for the membrane potential threshold while
`thrTheta`  is treated as a form of short-term plasticity (full
threshold is: `thr + thrTheta(t)`).

```{eval-rst}
.. autoclass:: ngclearn.components.LIFCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Quadratic LIF (Leaky Integrate-and-Fire) Cell

This cell (the quadratic "leaky integrator") models dynamics over the voltage
`v` and threshold shift `thrTheta` (a homeostatic variable). Note that
`thr` is used as a baseline level for the membrane potential threshold while
`thrTheta`  is treated as a form of short-term plasticity (full threshold
is: `thr + thrTheta(t)`). The dynamics are driven by a critical voltage value
as well as a voltage scaling factor for membrane potential accumulation over time.

```{eval-rst}
.. autoclass:: ngclearn.components.QuadLIFCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Adaptive Exponential (AdEx) Integrator Cell

This cell models dynamics over voltage `v` and a recover variable `w` (where `w`
governs the behavior of the action potential of a spiking neuronal cell). In
effect, the adaptive exponential (AdEx) integrate-and-fire model evolves as a
result of two coupled differential equations. (Note that this
cell supports either Euler or midpoint method / RK-2 integration.)

```{eval-rst}
.. autoclass:: ngclearn.components.AdExCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The FitzHugh–Nagumo Cell

This cell models dynamics over voltage `v` and a recover variable `w` (where `w`
governs the behavior of the action potential of a spiking neuronal cell). In
effect, the FitzHugh-Nagumo model is a set of two coupled differential equations
that simplify the four differential equation Hodgkin-Huxley (squid axon) model.
A voltage `v_thr` can be used to extract binary spike pulses. (Note that this
cell supports either Euler or midpoint method / RK-2 integration.)

```{eval-rst}
.. autoclass:: ngclearn.components.FitzhughNagumoCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Resonate-and-Fire (RAF) Cell

This cell models dynamics over voltage `v` and a angular driver state/variable `w`; these 
two variables result in a dampened oscillatory spiking neuronal cell). In effect, the 
resonatoe-and-fire (RAF) model (or "resonator") evolves as a result of two coupled 
differential equations. (Note that this cell supports either Euler or RK-2 integration.)

```{eval-rst}
.. autoclass:: ngclearn.components.RAFCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Izhikevich Cell

This cell models dynamics over voltage `v` and a recover variable `w` (where `w`
governs the behavior of the action potential of a spiking neuronal cell). In
effect, the Izhikevich model is a set of two coupled differential equations
that simplify the more complex dynamics of the Hodgkin-Huxley model. Note that
this Izhikevich model can be configured to model particular classes of neurons,
including regular spiking (RS), intrinsically bursting (IB), chattering (CH),
fast spiking (FS), low-threshold spiking (LTS), and resonator (RZ) neurons.
(Note that this cell supports either Euler or midpoint method / RK-2 integration.)

```{eval-rst}
.. autoclass:: ngclearn.components.IzhikevichCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### The Hodgkin-Huxley Cell

This cell models dynamics over voltage `v` and three channels/gates (related to 
potassium and sodium activation/inactivation). This sophisticated cell system is, 
as a result,  a set of four coupled differential equations and is driven by an appropriately  configured set of biophysical constants/coefficients (default values of which have  been set according to relevant source work). 
(Note that this cell supports either Euler or midpoint method / RK-2 integration.)

```{eval-rst}
.. autoclass:: ngclearn.components.HodgkinHuxleyCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```
