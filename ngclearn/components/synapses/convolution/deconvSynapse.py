from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
import ngclearn.utils.weight_distribution as dist

class DeconvSynapse(JaxComponent): ## static non-learnable synaptic cable
    """
    A static deconvolutional synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output
    | weights - current value tensor of kernel efficacies

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number input channels, number output channels, filter height,
            filter width)

        weight_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b
    """

    # Define Functions
    def __init__(self, name, shape, filter_init=None, bias_init=None,
                 Rscale=1., **kwargs):
        super().__init__(name, **kwargs)

        ## Synapse meta-parameters
        self.shape = shape ## shape of synaptic filter tensor
        self.Rscale = Rscale ## post-transformation scale factor

        tmp_key, *subkeys = random.split(self.key.value, 4)
        weights = None #dist.initialize_params(subkeys[0], filter_init, shape)

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.weights = Compartment(weights)