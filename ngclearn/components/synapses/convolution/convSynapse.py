from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.conv_utils import *

class ConvSynapse(JaxComponent): ## static non-learnable synaptic cable
    """
    A static convolutional synaptic cable; no form of synaptic evolution/adaptation
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

    # Define Functions
    def __init__(self, name, shape, x_size, filter_init=None, bias_init=None, stride=1,
                 padding=None, Rscale=1., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        ## Synapse meta-parameters
        self.shape = shape ## shape of synaptic filter tensor
        self.x_size = x_size
        self.Rscale = Rscale ## post-transformation scale factor
        self.padding = padding
        self.stride = stride

        ####################### Set up padding arguments #######################
        k_size, k_size, n_in_chan, n_out_chan = shape
        self.pad_args = None
        if self.padding is not None and self.padding == "SAME":
            if (x_size % stride == 0):
                pad_along_height = max(k_size - stride, 0)
            else:
                pad_along_height = max(k_size - (x_size % stride), 0)
            pad_bottom = pad_along_height // 2
            pad_top = pad_along_height - pad_bottom
            pad_left = pad_bottom
            pad_right = pad_top
            self.pad_args = ((pad_bottom, pad_top), (pad_left, pad_right))

        if self.padding is not None and self.padding == "VALID":
            self.pad_args = ((0, 0), (0, 0))

        ## set up compartments
        tmp_key, *subkeys = random.split(self.key.value, 4)
        weights = dist.initialize_params(subkeys[0], filter_init,
                                         shape)  ## filter tensor
        self.batch_size = batch_size # 1
        ## Compartment setup and shape computation
        _x = jnp.zeros((self.batch_size, x_size, x_size, n_in_chan))
        _d = conv2d(_x, weights, stride_size=stride, padding=padding) * 0
        self.in_shape = _x.shape
        self.out_shape = _d.shape
        self.inputs = Compartment(jnp.zeros(self.in_shape))
        self.outputs = Compartment(jnp.zeros(self.out_shape))
        self.weights = Compartment(weights)

        ########################################################################
        ## Shape error correction -- do shape correction inference
        '''
        _x = jnp.zeros((self.batch_size, x_size, x_size, n_in_chan))
        _d = conv2d(_x, self.weights.value, stride_size=self.stride,
                    padding=self.padding) * 0
        _dK = _calc_dK(_x, _d, stride_size=self.stride,
                       padding=self.pad_args)
        ## get filter update correction
        dx = _dK.shape[0] - self.weights.value.shape[0]
        dy = _dK.shape[1] - self.weights.value.shape[1]
        self.delta_shape = (dx, dy)
        ## get input update correction
        _dx = _calc_dX(self.weights.value, _d, stride_size=self.stride,
                       anti_padding=self.pad_args)
        dx = (_dx.shape[1] - _x.shape[1])
        dy = (_dx.shape[2] - _x.shape[2])
        self.x_delta_shape = (dx, dy)
        '''
        ########################################################################

    @staticmethod
    def _advance_state(padding, stride, weights, inputs):
        _x = inputs
        #print("in calc_out _x.shape: ", _x.shape)
        # if padding == "SAME":
        #     ((p1, p2), (p3, p4)) = get_same_conv_padding(_x, weights, stride_size=stride)
        #     jax_pad_args = ((p1.item(), p2.item()), (p3.item(), p4.item()))
        # elif padding == "VALID":
        #     ((p1, p2),(p3, p4)) = get_valid_conv_padding(_x, weights, stride_size=stride)
        #     jax_pad_args = ((p1.item(), p2.item()), (p3.item(), p4.item()))
        #print("in calc out : ", self.pad_args, jax_pad_args)
        return conv2d(_x, weights, stride_size=stride, padding=padding)

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _reset(in_shape, out_shape):
        preVals = jnp.zeros(in_shape)
        postVals = jnp.zeros(out_shape)
        inputs = preVals
        outputs = postVals
        return inputs, outputs

    @resolver(_reset)
    def reset(self, inputs, outputs):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
