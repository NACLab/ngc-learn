# GNCN-t1-Sigma (Friston, 2008)

This circuit implements the model proposed in (Friston, 2008) [1].
Specifically, this model is unsupervised and can be used to process sensory
pattern (row) vector(s) `x` to infer internal latent states. This class offers,
beyond settling and update routines, a projection function by which ancestral
sampling may be carried out given the underlying directed generative model
formed by this NGC system.

```{eval-rst}
.. autoclass:: ngclearn.museum.gncn_t1_sigma.GNCN_t1_Sigma
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
[1] Friston, Karl. "Hierarchical models in the brain." PLoS Computational
Biology 4.11 (2008): e1000211.
