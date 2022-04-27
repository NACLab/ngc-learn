# GNCN-t1-FFN (Whittington &amp; Bogacz, 2017)

This circuit implements the model proposed in ((Whittington &amp; Bogacz, 2017) [1].
Specifically, this model is supervised and can be used to process sensory
pattern (row) vector(s) `x` to predict target (row) vector(s) `y`. This class offers,
beyond settling and update routines, a prediction function by which ancestral
projection is carried out to efficiently provide label distribution or regression
vector outputs.

```{eval-rst}
.. autoclass:: ngclearn.museum.gncn_t1_ffn.GNCN_t1_FFN
  :noindex:

  .. automethod:: predict
    :noindex:
  .. automethod:: settle
    :noindex:
  .. automethod:: update
    :noindex:
  .. automethod:: clear
    :noindex:
```

**References:** <br>
[1] Whittington, James CR, and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.
