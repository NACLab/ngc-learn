# SNN-BA (Samadi et al., 2017)

This circuit implements the spiking neural model of (Samadi et al., 2017) [1].
Specifically, this model is **supervised** and can be used to process sensory
pattern (row) vector(s) `x` to predict target (row) vector(s) `y`. This class offers,
beyond settling and update routines, a prediction function by which ancestral
projection is carried out to efficiently provide label distribution or regression
vector outputs. Note that "SNN" denotes "spiking neural network" and "BA"
stands for "broadcast alignment". Note that it does not feature a separate
`calc_updates()` method like other models given that its `settle()` routine
adjusts synaptic efficacies dynamically (if configured to do so).

The SNN-BA is graphically depicted by the following graph:

```{eval-rst}
.. table::
   :align: center

   +-----------------------------------------------+
   | .. image:: ../images/museum/snn_ba.png        |
   |   :scale: 75%                                 |
   |   :align: center                              |
   +-----------------------------------------------+
```

```{eval-rst}
.. autoclass:: ngclearn.museum.snn_ba.SNN_BA
  :noindex:

  .. automethod:: predict
    :noindex:
  .. automethod:: settle
    :noindex:
  .. automethod:: clear
    :noindex:
```

**References:** <br>
[1] Samadi, Arash, Timothy P. Lillicrap, and Douglas B. Tweed. "Deep learning with
dynamic spiking neurons and fixed feedback weights." Neural computation 29.3
(2017): 578-602.
