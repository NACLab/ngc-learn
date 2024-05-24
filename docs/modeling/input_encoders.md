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

Input encoders generally take some data pattern(s) and transforms it to
another desired form, i.e., a real-valued vector to a sample of a spike train
at time `t`. Some input encoder components can emulate aspects of biological/
biophysical cells, e.g., the Poisson-distributed nature of the spikes emitted by
certain neuronal cells, or populations of them, e.g., Gaussian population
encoding.

### Bernoulli Cell

This cell takes a real-valued pattern(s) and transforms it on-the-fly to
a spike train where each spike vector is a sample of multivariate Bernoulli
distribution (the probability of a spike is proportional to the intensity of
the input vector value). NOTE: this assumes that each dimension of the
real-valued pattern vector is normalized between `[0,1]`.

```{eval-rst}
.. autoclass:: ngclearn.components.BernoulliCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Poisson Cell

This cell takes a real-valued pattern(s) and transforms it on-the-fly to
a spike train where each spike vector is a sample of Poisson spike-train with
a maximum frequency (given in Hertz).
NOTE: this assumes that each dimension of the real-valued pattern vector is
normalized between `[0,1]` (otherwise, results may not be as expected).

```{eval-rst}
.. autoclass:: ngclearn.components.PoissonCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Latency Cell

This cell takes a real-valued pattern(s) and transforms it on-the-fly to
a spike train where each spike vector is a sample of a latency (en)coded spike
train; higher intensity values will result in a firing time that occurs earlier
whereas lower intensity values yield later firing times.

```{eval-rst}
.. autoclass:: ngclearn.components.LatencyCell
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```
