# Metrics and Measurement Functions

Inside of `ngclearn.utils.metric_utils`, ngc-learn offers metrics and measurement
utility functions that can be quite useful when building neurocognitive models using
ngc-learn's node-and-cables system for specific tasks. While this utilities
sub-module will not always contain every possible function you might need,
given that measurements are often dependent on the task the experimenter wants
to conduct, there are several commonly-used ones drawn from machine intelligence
and computational neuroscience that are (jit-i-fied) in-built to ngc-learn you
can readily use.
In this small lesson, we will briefly examine two examples of importing such
functions and examine what they do.

## Measuring Task-Level Quantities

For many tasks that you might be interested in, a useful measurement
is the performance of the model in some supervised learning context. For example,
you might want to measure a model's accuracy on a classification task. To do so,
assuming we have some model outputs extracted from a model that you have constructed
elsewhere -- say a matrix of scores `Y_scores` -- and a target set of predictions
that you are testing against -- such as `Y_labels` (in one-hot binary encoded form )
-- then you can write some code to compute the accuracy, mean squared error (MSE),
and categorical log likelihood (Cat-NLL), like so:

```python
from jax import numpy as jnp
from ngclearn.utils.metric_utils import measure_ACC, measure_MSE, measure_CatNLL
from ngclearn.utils.model_utils import softmax

Y_scores = jnp.asarray([[5., -6., 12.],
                        [-11, 8., -2.],
                        [2., -1., 9.],
                        [15., 2.1, -32.],
                        [4., -11.2, -0.2]], dtype=jnp.float32)

Y_labels = jnp.asarray([[0., 0., 1.],
                        [0., 0., 1.],
                        [0., 1., 0.],
                        [1., 0., 0.],
                        [1., 0., 0.]], dtype=jnp.float32)

acc = measure_ACC(Y_scores, Y_labels)
mse = measure_MSE(Y_scores, Y_labels)
cnll = measure_CatNLL(softmax(Y_scores), Y_labels)

print(" > Accuracy = {:.3f}".format(acc))
print(" >      MSE = {:.3f}".format(mse))
print(" >  Cat-NLL = {:.3f}".format(cnll))
```
and you should obtain the following in I/O like so:

```console
 > Accuracy = 0.600
 >      MSE = 364.778
 >  Cat-NLL = 4.003
```

Notice that we imported the utility function `softmax` from
`ngclearn.utils.model_utils` to convert our raw theoretical model scores to
probability values so that using `measure_CatNLL()` makes sense (as this
assumes the model scores are normalized probability values).

## Measuring Spiking Model Statistics

In some cases, you might be interested in measuring certain statistics
related to aspects of a model that you construct. For example, you might have
collected a (binary) spike train produced by one of the internal neuronal layers
of your ngc-learn-simulated spiking neural network and want to compute the
firing rates and Fano factors associated with each neuron. Doing so with
ngc-learn utility functions would entail writing something like:

```python
from jax import numpy as jnp
from ngclearn.utils.metric_utils import measure_fanoFactor, measure_firingRate

spikes = jnp.asarray([[0., 0., 0.],
                      [0., 0., 1.],
                      [0., 1., 0.],
                      [1., 0., 1.],
                      [0., 0., 0.],
                      [0., 0., 1.],
                      [1., 0., 0.],
                      [0., 0., 1.],
                      [0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.],
                      [0., 0., 1.],
                      [0., 1., 0.],
                      [0., 0., 1.]], dtype=jnp.float32)

fr = measure_firingRate(spikes, preserve_batch=True)
fano = measure_fanoFactor(spikes, preserve_batch=True)

print(" > Firing Rates = {}".format(fr))
print(" > Fano Factor  = {}".format(fano))
```

which should result in the following to be printed to I/O:

```console
> Firing Rates = [[0.11111111 0.22222222 0.44444445]]
> Fano Factor  = [[0.8888888  0.77777773 0.55555546]]
```

The Fano factor is a useful secondary statistic for characterizing the
variable of a neuronal spike train -- as we see in the measurement above,
the first and second neurons have a higher Fano factor (given they are
more irregular in their spiking patterns) whereas the third neuron is far more
regular in its spiking pattern and thus has a lower Fano factor.
