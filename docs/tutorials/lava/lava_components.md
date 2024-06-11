# Lava Components

Inside ngc-learn, there is a wide variety of components with which biophysical 
models can be built. Unfortunately, many of those components are not compatible 
with Lava and the loihi2. Therefore, ngc-learn supports several in-built 
components that are Lava-compliant; many of the components that are compatible 
to you can be found in `ngclearn.components.lava`.

## What Makes an ngc-learn Component Compatible

For components to be compatible with Lava, there are a few key rules that must 
be followed:
- Lava Components can not make use of JAX's random or JAX's `nn` libraries
- Lava Components must import numpy from ngclearn and not JAX (there is a flag 
  in the configuration file to control JAX's numpy versus base numpy)
- Lava Components cannot take in any runtime arguments to their `advance_state` method
- Lava Components cannot take in any runtime arguments or compartments to their 
  `reset` method(s)

## Mapping Methods -- Going from ngc-learn to Lava

There are two methods that are mapped to their lava processes; these include the 
`reset` method and the `advance_state` method. The reset method is just mapped 
to the lava components and can be called on them without any issue. The
`advance_state` method is mapped to the `run_spk` method and is called during 
the runtime loops in Lava. It is important to note that the methods that are 
actually mapped are <i>the pure methods</i> passed into the resolvers that 
decorate the ngc-learn `reset` and `advance_state` methods, <b>not</b> the 
`reset` and `advance_state` methods themselves.