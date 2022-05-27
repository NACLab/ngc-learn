# Harmonium (Smolensky, 1986)

This circuit implements the Harmonium model proposed in (Smolensky, 1986) [1].
Specifically, this model is **unsupervised** and can be used to process sensory
pattern (row) vector(s) `x` to infer internal latent states. This class offers,
beyond settling and update routines through Contrastive Divergence (Hinton 1999) [2],
a block Gibbs sampling function to generate a chain of synthesized patterns.

The Harmonium is technically defined by two NGC graphs. The first is the positive
phase ("wake" phase) graph depicted graphically below:

```{eval-rst}
.. table::
   :align: center

   +-----------------------------------------------+
   | .. image:: ../images/museum/rbm_pos.png       |
   |   :scale: 75%                                 |
   |   :align: center                              |
   +-----------------------------------------------+
```

while second is the negative phase ("sleep" phase) graph depicted graphically below:

```{eval-rst}
.. table::
   :align: center

   +-----------------------------------------------+
   | .. image:: ../images/museum/rbm_neg.png       |
   |   :scale: 75%                                 |
   |   :align: center                              |
   +-----------------------------------------------+
```


```{eval-rst}
.. autoclass:: ngclearn.museum.harmonium.Harmonium
  :noindex:

  .. automethod:: sample
    :noindex:
  .. automethod:: settle
    :noindex:
  .. automethod:: calc_updates
    :noindex:
  .. automethod:: clear
    :noindex:
```

**References:** <br>
[1] Smolensky, P. "Information Processing in Dynamical Systems: Foundations of
Harmony Theory." Parallel distributed processing: explorations in the
microstructure of cognition 1 (1986).<br>
[2] Hinton, Geoffrey E. "Training products of experts by maximizing contrastive
likelihood." Technical Report, Gatsby computational neuroscience unit (1999).
