# Model

This is the base class/construct from which biomimetic computational models
should inherit from/extend. Only simple base-level functionality is required,
i.e., a `save(.)` and `load(.)` routine, beyond the base constructor. A `Model`
typically encapsulates one or more `OpGraph` constructs and instantiates its
own model-specific, task-specific sub-routines (typically needed for
reproducing an experimental result or figure found in its source paper)

Any official historical model that is slated to be integrated into the ngc-learn
Model Museum should first sub-class the `Model` base class and ensure that very
basic arguments such `config` are available/provided. Documentation of the model's
source publication/paper (in APA or MLA format) must be provided in the model's
official doc-string.

Basic format/functionality of a `Model` is:
```{eval-rst}
.. autoclass:: ngclearn.museum.model.Model
  :noindex:

  .. automethod:: save
    :noindex:
  .. automethod:: load
    :noindex:
```
