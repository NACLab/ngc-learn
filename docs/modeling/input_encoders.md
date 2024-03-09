# Input Encoders

The input encoder is generally a utility component for encoding sensory input /
data in a certain way. A typical use-case for an input encoder in the context
of spiking neural networks is to integrate a non-parameterized transformation
into a system that, ideally on-the-fly, transforms a input pattern(s) to a
spike trains -- for example, one can use a [PoissonCell](ngclearn.components.input_encoders.poissonCell)
to transform a fixed input into an approximate spike train with a maximum
frequency (in Hertz). Some of these encoders, such as the Bernoulli and
Poisson cells, can be interpreted as coarse-grained approximations of dynamics
that would normally be modeled by complex differential equations.

## Input Encoding Operator Types

### Bernoulli Cell

```{eval-rst}
.. autoclass:: ngclearn.components.BernoulliCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Poisson Cell

```{eval-rst}
.. autoclass:: ngclearn.components.PoissonCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```
