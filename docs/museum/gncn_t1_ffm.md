# GNCN-t1-FFM (Whittington &amp; Bogacz, 2017)

This circuit implements the model proposed in ((Whittington &amp; Bogacz, 2017) [1].
Specifically, this model is **supervised** and can be used to process sensory
pattern (row) vector(s) `x` to predict target (row) vector(s) `y`. This class offers,
beyond settling and update routines, a prediction function by which ancestral
projection is carried out to efficiently provide label distribution or regression
vector outputs. Note that "FFM" denotes "feedforward mapping".

The GNCN-t1-FFM is graphically depicted by the following graph:

```{eval-rst}
.. table::
   :align: center

   +---------------------------------------------------+
   | .. image:: ../images/museum/gncn_t1_ffm.png       |
   |   :scale: 75%                                     |
   |   :align: center                                  |
   +---------------------------------------------------+
```

```{eval-rst}
.. autoclass:: ngclearn.museum.gncn_t1_ffm.GNCN_t1_FFM
  :noindex:

  .. automethod:: predict
    :noindex:
  .. automethod:: settle
    :noindex:
  .. automethod:: calc_updates
    :noindex:
  .. automethod:: clear
    :noindex:
```

**References:** <br>
[1] Whittington, James CR, and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.
