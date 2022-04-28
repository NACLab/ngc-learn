# GNCN-PDH (Ororbia &amp; Kifer, 2020/2022)

This circuit implements one of the models proposed in (Ororbia &amp; Kifer, 2022) [1].
Specifically, this model is **unsupervised** and can be used to process sensory
pattern (row) vector(s) `x` to infer internal latent states. This class offers,
beyond settling and update routines, a projection function by which ancestral
sampling may be carried out given the underlying directed generative model
formed by this NGC system.

```{eval-rst}
.. autoclass:: ngclearn.museum.gncn_pdh.GNCN_PDH
  :noindex:

  .. automethod:: project
    :noindex:
  .. automethod:: settle
    :noindex:
  .. automethod:: update
    :noindex:
  .. automethod:: clear
    :noindex:
```

**References:** <br>
[1] Ororbia, A., and Kifer, D. The neural coding framework for learning
generative models. Nature Communications 13, 2064 (2022).
