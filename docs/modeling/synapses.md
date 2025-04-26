# Synapses

The synapse is a key building block for connecting/wiring together the various
component cells that one would use for characterizing a biomimetic neural system.
These particular objects are meant to perform, per simulated time step, a
specific type of transformation -- such as a linear transform or a 
convolution -- utilizing their underlying synaptic parameters.
Most times, a synaptic cable will be represented by a set of matrices (or filters) 
that are used to conduct a projection of an input signal (a value presented to a
pre-synaptic/input compartment) resulting in an output signal (a value that
appears within one of its post-synaptic compartments). Notably, a synapse component is
typically associated with a local plasticity rule, e.g., a Hebbian-type
update, that either is triggered online, i.e., at some or all simulation time
steps, or by integrating a differential equation, e.g., via eligibility traces.

## Non-Plastic Synapse Types

### Static (Dense) Synapse

This synapse performs a linear transform of its input signals. Note that this synaptic cable does not evolve and is meant to be used for fixed value (dense) synaptic connections. Note that all synapse components that inherit from the `DenseSynapse` class support sparsification (via the `p_conn` probability of existence argument) to produce sparsely-connected synaptic connectivity patterns. 

```{eval-rst}
.. autoclass:: ngclearn.components.StaticSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Static Convolutional Synapse

This synapse performs a convolutional transform of its input signals.
Note that this synaptic cable does not evolve and is meant to be used for fixed value convolution synaptic filters.

```{eval-rst}
.. autoclass:: ngclearn.components.ConvSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Static Deconvolutional Synapse

This synapse performs a deconvolutional transform of its input signals. Note that this synaptic cable does not evolve and is meant to be used for fixed value deconvolution/transposed convolution synaptic filters.

```{eval-rst}
.. autoclass:: ngclearn.components.DeconvSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

## Dynamic Synapse Types

### Exponential Synapse

This (chemical) synapse performs a linear transform of its input signals. Note that this synapse is "dynamic" in the sense that its efficacies are a function of their pre-synaptic inputs; there is no inherent form of long-term plasticity in this base implementation. Synaptic strength values can be viewed as being filtered/smoothened through an expoential kernel. 

```{eval-rst}
.. autoclass:: ngclearn.components.ExponentialSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Alpha Synapse

This (chemical) synapse performs a linear transform of its input signals. Note that this synapse is "dynamic" in the sense that its efficacies are a function of their pre-synaptic inputs; there is no inherent form of long-term plasticity in this base implementation. Synaptic strength values can be viewed as being filtered/smoothened through a kernel that models more realistic rise and fall times of synaptic conductance..

```{eval-rst}
.. autoclass:: ngclearn.components.AlphaSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Short-Term Plasticity (Dense) Synapse

This synapse performs a linear transform of its input signals. Note that this synapse is "dynamic" in the sense that it engages in short-term plasticity (STP), meaning that its efficacy values change as a function of its inputs/time (and simulated consumed resources), but it does not provide any long-term form of plasticity/adjustment.

```{eval-rst}
.. autoclass:: ngclearn.components.STPDenseSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: reset
    :noindex:
```

## Multi-Factor Learning Synapse Types

Hebbian rules operate in a local manner -- they generally use information more immediately available to synapses in both space and time -- and can come in a wide variety of flavors. One general way to categorize variants of Hebbian learning is to clarify what (neural) statistics/values they operate on, e.g, do they work with real-valued information or discrete spikes, and how many factors (or distinct terms) are involved in calculating the update to synaptic values by the relevant learning rule. <!--(Note that, in principle, all forms of plasticity in ngc-learn technically work like local, factor-based rules. )-->

### (Two-Factor) Hebbian Synapse

This synapse performs a linear transform of its input signals and evolves according to a strictly two-factor/term update rule. In other words, the underlying synaptic efficacy matrix is changed according to a product between pre-synaptic compartment values (`pre`) and post-synaptic compartment (`post`) values, which can contain any type of vector/matrix statistics. This particular synapse further features some tools for advanced forms of Hebbian descent/ascent (such as applying the Hebbian update via adaptive learning rates such as adaptive moment estimation, i.e., Adam).

```{eval-rst}
.. autoclass:: ngclearn.components.HebbianSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

### (Two-Factor) BCM Synapse

This synapse performs a linear transform of its input signals and evolves according to a multi-factor, Bienenstock-Cooper-Munro (BCM) update rule. The underlying synaptic efficacy matrix is changed according to an evolved synaptic threshold parameter `theta` and a product between pre-synaptic compartment values (`pre`) and a nonlinear function of post-synaptic compartment (`post`) values, which can contain any type of vector/matrix statistics.

```{eval-rst}
.. autoclass:: ngclearn.components.BCMSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

### (Two-Factor) Hebbian Convolutional Synapse

This synapse performs a convolutional transform of its input signals and evolves according to a two-factor update rule. The underlying synaptic filters are changed according to products between pre-synaptic compartment values (`pre`) and post-synaptic compartment (`post`)  feature map values.

```{eval-rst}
.. autoclass:: ngclearn.components.HebbianConvSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

### (Two-Factor) Hebbian Deconvolutional Synapse

This synapse performs a deconvolutional (transposed convolutional) transform of its input signals and evolves according to a two-factor update rule. The underlying synaptic filters are changed according to products between pre-synaptic compartment values (`pre`) and post-synaptic compartment (`post`) feature map values.

```{eval-rst}
.. autoclass:: ngclearn.components.HebbianDeconvSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

## Spike-Timing-Dependent Plasticity (STDP) Synapse Types

Synapses that evolve according to a spike-timing-dependent plasticity (STDP) process operate, at a high level, much like multi-factor Hebbian rules (given that STDP is a generalization of Hebbian adjustment to spike trains) and share many of their properties. Nevertheless, a distinguishing factor for STDP-based synapses is that they must involve action potential pulses (spikes) in their calculations and they typically compute synaptic change according to the relative timing of spikes. In principle, any of the synapses in this grouping of components adapt their efficacies according to rules that are at least special four terms, i.e., a pre-synaptic spike (an "event"), a pre-synaptic delta timing (which can come in the form of a trace), a post-synaptic spike (or event), and a post-synaptic delta timing (also can be a trace). In addition, STDP rules in ngc-learn typically enforce soft/hard synaptic strength bounding, i.e., there is a maximum magnitude allowed for any single synaptic efficacy, and, by default, an STDP synapse enforces its synaptic strengths to be non-negative. 
Note: these rules are technically considered to be "two-factor" rules since they only operate on pre- and post-synaptic activity (despite each factor being represented by two or more terms).

### Trace-based STDP

This is a four-term STDP rule that adjusts the underlying synaptic strength matrix via a weighted combination of long-term depression (LTD) and long-term potentiation (LTP). For the LTP portion of the update, a pre-synaptic trace and a post-synaptic event/spike-trigger are used, and for the LTD portion of the update, a pre-synaptic event/spike-trigger and a post-synaptic trace are utilized. Note that this specific rule can be configured to use different forms of soft threshold bounding including a scheme that recovers a power-scaling form of STDP (via the hyper-parameter `mu`).

```{eval-rst}
.. autoclass:: ngclearn.components.TraceSTDPSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Exponential STDP

This is a four-term STDP rule that directly incorporates a controllable exponential synaptic strength dependency into its dynamics. This synapse's LTP and LTD use traces and spike events in a manner similar to the trace-based STDP described above.

```{eval-rst}
.. autoclass:: ngclearn.components.ExpSTDPSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Event-Driven Post-Synaptic STDP Synapse

This is a synaptic evolved under a two-term STDP rule that is driven by only spike events and operates within a defined pre-synaptic "window" of time.

```{eval-rst}
.. autoclass:: ngclearn.components.EventSTDPSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Trace-based STDP Convolutional Synapse

This is a four-term STDP rule for convolutional synapses/kernels that adjusts the underlying filters via a weighted combination of long-term depression (LTD) and long-term potentiation (LTP). For the LTP portion of the update, a pre-synaptic trace and a post-synaptic event/spike-trigger are used, and for the LTD portion of the update, a pre-synaptic event/spike-trigger and a post-synaptic trace are utilized. 

```{eval-rst}
.. autoclass:: ngclearn.components.TraceSTDPConvSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

### Trace-based STDP Deonvolutional Synapse

This is a four-term STDP rule for deconvolutional (transposed convolutional) synapses that adjusts the underlying filters via a weighted combination of long-term depression (LTD) and long-term potentiation (LTP). For the LTP portion of the update, a pre-synaptic trace and a post-synaptic event/spike-trigger are used, and for the LTD portion of the update, a pre-synaptic event/spike-trigger and a post-synaptic trace are utilized. 

```{eval-rst}
.. autoclass:: ngclearn.components.TraceSTDPDeconvSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```

## Modulated Forms of Plasticity

This family of synapses implemented within ngc-learn support modulated, often 
at least three-factor, forms of synaptic adjustment. Modulators could include reward/dopamine values or scalar error signals, and are generally assumed to be administered to the synapse(s) externally (i.e., it is treated as another input provided by some other entity, e.g., another neural circuit).

### Reward-Modulated Trace-based STDP (MSTDP-ET)

This is a three-factor learning rule, i.e., pre-synaptic activity, post-synaptic activity, and a modulatory signal, known as modulated STDP (MSTDP). MSTDP adjusts the underlying synaptic strength matrix via a weighted combination of long-term depression (LTD) and long-term potentiation (LTP), scaled by an external signal such as a reward/dopamine value. The STDP element of this form of plasticity inherits from trace-based STDP (i.e. `TraceSTDPSynapse`). Note that this synapse component further supports a configuration for MSTDP-ET, or MSTPD with eligibility traces; MSTDP-ET means that the synapse will treat its synapses as two elements -- a synaptic efficacy and a coupled synaptic trace that maintains the dynamics of STDP updates encountered over time.

```{eval-rst}
.. autoclass:: ngclearn.components.MSTDPETSynapse
  :noindex:

  .. automethod:: advance_state
    :noindex:
  .. automethod:: evolve
    :noindex:
  .. automethod:: reset
    :noindex:
```
