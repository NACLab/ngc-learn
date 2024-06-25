from .convSynapse import ConvSynapse

class StaticConvSynapse(ConvSynapse):
    """
    A static convolutional synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals
    | weights - current value tensor of kernel efficacies
    | biases - current base-rate/bias efficacies

    Args:
        name: the string name of this cell

        x_shape: 2d shape of input map signal (component currently assumess a square input maps)

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number `filter height x filter width x input channels x number output channels`);
            note that currently filters/kernels are assumed to be square
            (kernel.width = kernel.height)

        filter_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values
            (Default: None, which turns off/disables biases)

        stride: length/size of stride

        padding: pre-operator padding to use -- "VALID" (none), "SAME"

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((K @ in) * resist_scale) + b
            where `@` denotes convolution

        batch_size: batch size dimension of this component
    """

    pass
