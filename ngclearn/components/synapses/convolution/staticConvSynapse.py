from .convSynapse import ConvSynapse

class StaticConvSynapse(ConvSynapse):
    """
    A base convolutional synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output
    | weights - current value tensor of kernel efficacies

    Args:
        name: the string name of this cell

        x_size: dimension of input signal (assuming a square input)

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number input channels, number output channels, filter height,
            filter width)

        weight_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b
    """

    pass