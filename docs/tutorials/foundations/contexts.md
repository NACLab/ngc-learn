# What are Contexts

A context in ngclearn is a container that holds all the information for your model and can be used as an access point to
reference different models in a multi-model system. Some of the information that contexts hold is all the components
defined in the context, all the wiring information for each of the components, as well as all the commands defined on
the context through various means.

## How to make a Context

To make a context first import it from ngclearn with `from ngclearn import Context`. This will give you access to not
only the constructor for new contexts but also the ability to get previously defined contexts. The general use case for
this is

```python
from ngclearn import Context

with Context("Model1") as model1:
    pass
```

This will make a context named "Model1" and also drops you into a with block where you can define the various parts of
the model. The call `Context("Model1")` will always return the same context. So if there is already a model with that
name defined earlier in the code this instance of `model1` will have all the same object defined previously.

## Adding Components

The best way to add components to a context is by using components that have implemented the `MetaComponent` metaclass.
In ngclearn the base `Component` class does this. If using these components all that is needed to have them added to
the context is calling their constructors inside a with block of the context. For example

```python
from ngclearn import Context
from ngclearn.components import LIFCell

with Context("Model1") as model1:
    z1 = LIFCell("z1", n_units=10, tau_m=100)
```

## Creating Cables

To add connections between components and their compartments in a model we do that also in a context. Just like with
components there are no special actions that need to be taken to add them beyond doing so in a with block. To connect
to compartments the `<<` operator is used following the outline of `destination << source`. For example

```python
with model1:
    w1.inputs << z1.s
```

## Dynamic Commands

When building models it can be desirable to use the same training and testing scrips while having commands do different
actions. For example if two different models had different clamp procedures to set inputs and labels it is possible to
dynamically add a generic clamp command to each model and call them the same way despite them doing different things.
As an example
```python
with model1:
    @model1.dynamicCommand
    def clamp(inputs, labels):
        z0.inputs.clamp(inputs)
        z2.labels.clamp(labels)

with model2:
    @model2.dynamicCommand
    def clamp(inputs, labels):
        z0.inputs.clamp(inputs)
        z0_p.inputs.clamp(inputs)
        z2.labels.clamp(labels)
```
In both these cases later we can just call clamp and each one will call their own version of the clamp command.
