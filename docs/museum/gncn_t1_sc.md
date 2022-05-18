# GNCN-t1-SC (Olshausen &amp; Field, 1996)

This circuit implements the sparse coding model proposed in (Olshausen &amp; Field, 1996) [1].
Specifically, this model is **unsupervised** and can be used to process sensory
pattern (row) vector(s) `x` to infer internal latent states. This class offers,
beyond settling and update routines, a projection function by which ancestral
sampling may be carried out given the underlying directed generative model
formed by this NGC system.

```{eval-rst}
.. autoclass:: ngclearn.museum.gncn_t1_sc.GNCN_t1_SC
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
[1] Olshausen, B., Field, D. Emergence of simple-cell receptive field properties
by learning a sparse code for natural images. Nature 381, 607–609 (1996).
