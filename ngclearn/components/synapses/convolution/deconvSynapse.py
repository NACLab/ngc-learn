from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.conv_utils import *

#from ngclearn.utils.conv_utils import _deconv_same_transpose_padding, _deconv_valid_transpose_padding
from ngclearn.utils.conv_utils import conv2d, deconv2d, rot180

class DeconvSynapse(JaxComponent): ## static non-learnable synaptic cable
    """
    A base deconvolutional synaptic cable.

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

        ## set up compartments
        tmp_key, *subkeys = random.split(self.key.value, 4)
        weights = dist.initialize_params(subkeys[0], filter_init,
                                         shape)  ## filter tensor
        self.batch_size = batch_size # 1
        ## Compartment setup and shape computation
        _x = jnp.zeros((self.batch_size, x_size, x_size, n_in_chan))
        _d = deconv2d(_x, weights, stride_size=stride, padding=padding) * 0
        self.in_shape = _x.shape
        self.out_shape = _d.shape
        self.inputs = Compartment(jnp.zeros(self.in_shape))
        self.outputs = Compartment(jnp.zeros(self.out_shape))
        self.weights = Compartment(weights)

        ########################################################################
        ## Shape error correction -- do shape correction inference (for local updates)
        '''
        _x = jnp.zeros((batch_size, x_size, x_size, n_in_chan))
        _d = deconv2d(_x, self.K, stride_size=self.stride, padding=self.padding) * 0
        _dK = _calc_dK(_x, _d, stride_size=self.stride, out_size=self.k_size)
        ## get filter update correction
        dx = _dK.shape[0] - self.K.shape[0]
        dy = _dK.shape[1] - self.K.shape[1]
        self.delta_shape = (abs(dx), abs(dy))

        ## get input update correction
        _dx = _calc_dX(self.K, _d, stride_size=self.stride, padding = self.padding)
        dx = (_dx.shape[1] - _x.shape[1]) # abs()
        dy = (_dx.shape[2] - _x.shape[2])
        self.x_delta_shape = (dx, dy)
        '''
        ########################################################################

    @staticmethod
    def _advance_state(padding, stride, weights, inputs):
        _x = inputs
        out = deconv2d(_x, weights, stride_size=stride, padding=padding)
        return out

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
