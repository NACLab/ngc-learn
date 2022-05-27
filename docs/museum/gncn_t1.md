# GNCN-t1 (Rao &amp; Ballard, 1999)

This circuit implements the model proposed in (Rao &amp; Ballard, 1999) [1].
Specifically, this model is **unsupervised** and can be used to process sensory
pattern (row) vector(s) `x` to infer internal latent states. This class offers,
beyond settling and update routines, a projection function by which ancestral
sampling may be carried out given the underlying directed generative model
formed by this NGC system.

The GNCN-t1 is graphically depicted by the following graph:

```{eval-rst}
.. table::
   :align: center

   +-----------------------------------------------+
   | .. image:: ../images/museum/gncn_t1.png       |
   |   :scale: 75%                                 |
   |   :align: center                              |
   +-----------------------------------------------+
```

```{eval-rst}
.. autoclass:: ngclearn.museum.gncn_t1.GNCN_t1
  :noindex:

  .. automethod:: project
    :noindex:
  .. automethod:: settle
    :noindex:
  .. automethod:: calc_updates
    :noindex:
  .. automethod:: clear
    :noindex:
```

**References:** <br>
[1] Rao, Rajesh PN, and Dana H. Ballard. "Predictive coding in the visual
cortex: a functional interpretation of some extra-classical receptive-field
effects." Nature neuroscience 2.1 (1999): 79-87.
