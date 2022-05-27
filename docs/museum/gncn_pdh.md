# GNCN-PDH (Ororbia &amp; Kifer, 2020/2022)

This circuit implements one of the models proposed in (Ororbia &amp; Kifer, 2022) [1].
Specifically, this model is **unsupervised** and can be used to process sensory
pattern (row) vector(s) `x` to infer internal latent states. This class offers,
beyond settling and update routines, a projection function by which ancestral
sampling may be carried out given the underlying directed generative model
formed by this NGC system.

The GNCN-PDH is graphically depicted by the following graph:

```{eval-rst}
.. table::
   :align: center

   +------------------------------------------------+
   | .. image:: ../images/museum/gncn_pdh.png       |
   |   :scale: 75%                                  |
   |   :align: center                               |
   +------------------------------------------------+
```

```{eval-rst}
.. autoclass:: ngclearn.museum.gncn_pdh.GNCN_PDH
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
[1] Ororbia, A., and Kifer, D. The neural coding framework for learning
generative models. Nature Communications 13, 2064 (2022).
