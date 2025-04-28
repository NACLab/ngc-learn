# Lecture 4B: Hebbian Synaptic Plasticity

In ngc-learn, synaptic plasticity is a key concept at the forefront of its
design in order to promote research into novel ideas and framings of how
adaptation might occur at various time-scales in biomimetic systems such as
those composed of neurons. One of the simplest and most useful ways that one
may implement plasticity is through a scheme that embodies principles of
Hebbian learning -- characterized by the well-known popularized phrase
"neurons that fire together, wire together" <b>[1]</b>[^1].

In the [evolving synapse lesson](../model_basics/evolving_synapses.md) under
"Model Basics", we cover how one would construct a basic two-factor Hebbian rule
for adjusting the values of a synapse connecting two `RateCell` cell components.
For the purposes of this lesson, we will just point out a key points of the
`HebbianSynapse`, found within `ngclearn.components.synapses.hebbian.hebbianSynapse`,
as this synaptic component highlights a few programmatic elements unique to how
synapses operate within ngc-learn's nodes-and-cables system.

Specifically, we will zoom in on two particular code snippets from
[evolving synapses tutorial](../model_basics/evolving_synapses.md), reproduced
below:

```python
Wab = HebbianSynapse(name="Wab", shape=(1, 1), eta=1.,   signVal=-1.,
                     wInit=("constant", 1., None), w_bound=0., key=subkeys[3])

# wire output compartment (rate-coded output zF) of RateCell `a` to input compartment of HebbianSynapse `Wab`
Wab.inputs << a.zF
# wire output compartment of HebbianSynapse `Wab` to input compartment (electrical current j) RateCell `b`
b.j << Wab.outputs

# wire output compartment (rate-coded output zF) of RateCell `a` to presynaptic compartment of HebbianSynapse `Wab`
Wab.pre << a.zF
# wire output compartment (rate-coded output zF) of RateCell `b` to postsynaptic compartment of HebbianSynapse `Wab`
Wab.post << b.zF
```

as well as (a bit later in the model construction code):

```python
evolve_process = (JaxProcess()
                  >> a.evolve)
circuit.wrap_and_add_command(jit(evolve_process.pure), name="evolve")

advance_process = (JaxProcess()
                   >> a.advance_state)
circuit.wrap_and_add_command(jit(advance_process.pure), name="advance")
```

Notice that beyond wiring component `a`'s values into the synapse `Wab`'s input compartment
`Wab.inputs` (and wiring `Wab`'s output compartment `Wab.outputs` to node `b`), we
also wired values to `Wab`'s pre-synaptic compartment `Wab.pre` and `Wab`'s
post-synaptic compartment `Wab.post`. These compartments are specifically
used in `Wab`'s `evolve` call and are not strictly required to be exactly
the same as its input and output compartments. Note that, if one wanted `pre`
and `post` to be exactly identical to `inputs` and `outputs`, one would simply need
to write `Wab.pre << Wab.inputs` and `Wab.post << Wab.outputs` in place
of the pre- and post-synaptic compartment calls above.

The above snippets highlight two key aspects of functionality that a synapse
entails as opposed to the cell components (like `RateCell` or `AdExCell`):
1. Synapse components contain additional compartments that specifically relate
to their long-term evolution/adaptation, i.e., their "learning" ability, that
might be directly the same as their input and output compartments;
2. The `evolve` command is specifically used by synapse components, as this
command triggers a different set of dynamics that might be the same as those
entailed by `advance_state` (meaning that a call to evolve for synapse components
entails a possibly different time-scale of adaptation/evolution).

Technically, all components in ngc-learn inherit the ability to run an `advance_state`
as well as an `evolve`, but most only use `advance_state` with the notable
exception of synapse components (which, as seen above, make use of both). This
particular separation of time-scales with the `advance_state` and `evolve` are
important for two reasons:
1. synapses might be updated at very specific time steps within a simulation as
"events" rather than being run continuously as cell components typically
are, i.e., in other words, a call to a synapse component's `evolve` might only
happen a few times within a window of time (and we wish to save on calls to
the computation underlying `evolve` that are not needed to speed up our
simulations greatly);
2. a call to a synapse's `evolve` might involve statistics that are not
available every time step (or are not meant to be visible every time step).
A good modeling use-case that embodies the above two reasons is in predictive coding
or sparse coding/dictionary learning models -- these kinds of models are often
formulated from the perspective of expectation-maximization (EM) where the activity
values of neuronal layers are evolved continuously over a window of time (the
E-step) and then synaptic efficacies are adjusted based on the final state of the
layer-wise activities at the end of this window (the M-step). One does not want
to waste the call to an M-step after each E-step (since only the M-step at the
end of several E-steps is of interest) and, furthermore, the inputs to the
M-step (which would be the compartment values involved in the `evolve` call
of a synapse component) involve values beyond just the exact input and output
of the synapse itself (usually the M-step involves using the input to the
synapse and the activity values of nearby error neurons).
In the model museum, we see explicitly how the EM adaptation scheme of a predictive
coding circuit is implemented in [the walkthrough](../../museum/pcn_discrim.md) and
the corresponding Github code
[PCN model](https://github.com/NACLab/ngc-museum/tree/main/exhibits/pc_discrim).

A final note with respect to `evolve` and `advance_state` is that one does not
have to use or implement `evolve` if one wants synaptic connection update to occur
exactly within a synapse component's `advance_state` (say, in a custom
component that the experimenter decides to write for their work); the only tricky part of
implementing a change in synaptic efficacy directly inside of `advance_state` is
that the designer would need to ensure that all statistics needed for changing the
synaptic component's internal values are available in the synapse's compartments
exactly each time the `advance_state` is called (at the exact same simulation
time step that, for example, that forwarding of signals across a synapse occurs).
This would possibly be appropriate for synapses that frame their change in
synaptic efficacies in terms of differential equations that need to be clocked
in the exact same way as the dynamics of the relevant neuronal cell components
(though the same effect can be emulated with a pairing of calls to the
component's `advance_state` and `evolve`).

## References
<b>[1]</b> Donald, Hebb. "The organization of behavior." Wiley: New York (1949).

<!-- Footnotes -->
[^1]: The actual statement made by Donald Hebb was: "When an axon of cell A is
near enough to excite a cell B and repeatedly or persistently takes part in
firing it,...A’s efficiency, as one of the cells firing B, is increased".
