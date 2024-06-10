# Lava Components

Inside ngc-learn there is a wide variety of components with which models can be built. Unfortunately, many of those
components are not compatible with lava and the loihi2. Therefor many of the components that are compatible can be found
in `ngclearn.components.lava`.

## What makes it compatible

For components to be compatible with lava there are a few key rules that must be followed.

- Lava Components can not make use of JAX's random or JAX's nn libraries
- Lava Components must import numpy from ngclearn not JAX (There is a flag in the configuration file to control JAX's
  numpy vs base numpy)
- Lava Components can not take in any runtime arguments to their advance_state method
- Lava Components can not take in any runtime arguments or compartment to their reset methods

## Mapping methods

There are two methods that are mapped to their lava processes these are the `reset` method and the `advance_state`
method. The reset method is just mapped to the lava components and can be called on them without issue. The
advance_state method is mapped to the `run_spk` method and is called during the runtime loops in lava. It is important
to note that the method that are actually mapped are the pure method passed into the resolvers decorating the reset and
advance_state methods, not the reset and advance_state methods themselves.