# Other Operators

Other operators range from variable traces to kernels and hand-crafted transformations.
An important and oft-used one, in the case of spiking neural systems, is the
variable trace (or filter) -- for instance, one might need to track a cumulative
value based on spikes over time to trigger local updates to synaptic cable values
and a compartment such as [VarTrace](ngclearn.components.other.varTrace).

## Trace Operators

### Variable Trace

```{eval-rst}
.. autoclass:: ngclearn.components.VarTrace
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: verify_connections
    :noindex:
  .. automethod:: reset
    :noindex:
```
