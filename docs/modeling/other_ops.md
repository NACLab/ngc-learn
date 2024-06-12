# Other Operators

Other operators range from variable traces to kernels and hand-crafted transformations.
An important and oft-used one, in the case of spiking neural systems, is the
variable trace (or filter) -- for instance, one might need to track a cumulative
value based on spikes over time to trigger local updates to synaptic cable values
with a compartment such as [VarTrace](ngclearn.components.other.varTrace).

## Trace Operators

### Variable Trace

This operator processes and tracks a particular value (dependent upon which
external component's compartment is wired into this one's input compartment).  
In general, a trace integrates a differential equation based on an external
component's compartment value, e.g., the spike `s` of a spiking neuronal cell,
producing a real-valued cumulative representation of it across time. For
instance, instead of directly tracking spike times of a particular spiking cell,
a trace can be used to represent a soft, single approximation. Another way to
view a variable trace is that it acts as a low-pass filter of another signal
sequence.

```{eval-rst}
.. autoclass:: ngclearn.components.VarTrace
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

## Kernels

Kernels are an important and useful building block for constructing what is
known in computational neuroscience as spike-response models (SRMs). In
ngc-learn, these generally involve the construction of nodes that apply a
particular mathematical function (or set of them) to integrate over a window
of collected values, generally discrete spikes or action potentials produced
within a particular window of time.

### Exponential Kernel

This kernel operator processes and tracks a window of values (generally spikes) to
produce an excitatory postsynaptic potential (EPSP) pulse value via application
of an exponential kernel.

```{eval-rst}
.. autoclass:: ngclearn.components.ExpKernel
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```
