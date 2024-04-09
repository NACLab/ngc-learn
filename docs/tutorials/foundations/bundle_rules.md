# The Basics of Bundle Rules

## What are Bundle Rules?
The underlying method for passing data/signals from component to component inside of
controllers is through the use of cables. Most of the time, each compartment of a
component will have a single incoming connection, but sometimes there will be
more than one; we call this a "bundle of cables". A bundle rule is a method that
defines how each cable (within a bundle) will interact with the destination
compartment.

## Built-in Rules
By default, ngcsimlib comes with three bundle rules,
`overwrite`, `additive`, and `append`. Of these rules, the default rule for every
cable is overwrite, which replaces the value of the compartment with the value
being transmitted along the cable. The additive compartment rule itself adds to
the existing value while the append compartment calls the append method on the
compartment with the transmitted value.

## Building Custom Rules
At its core, a bundle rule is simply a method with three parameters. The first
parameter is the destination component, the second is the value being
transmitted, and finally the destination compartment. A bundle rule has no
return value.

> General Form of a Bundle Rule Routine:
> ```python
> def BUNDLE_RULE_NAME(component, value, destination_compartment):
>     # Logic for processing transmitted value
>     # Syntax for referencing destination compartment ->
>         # component.compartments[destination_compartment]
>     pass
> ```

> Example Bundle Rule (Additive)
> ```python
> def additive(component, value, destination_compartment):
>     component.compartments[destination_compartment] += value
> ```

## Notes
Each cable can have a different bundle rule, so the order that connections are
instantiated/drawn potentially will have an impact on the final value of a
compartment.

In the case of a rule such as additive, consider looking at the `pregather`
method found in the base component. This routine can be used to reset states
prior to all of the connections/cable transmissions being computed.

Note that bundle rules can be used even if there is only one value connected to a
compartment.
