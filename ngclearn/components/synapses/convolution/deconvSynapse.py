from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.conv_utils import *

from ngclearn.utils.conv_utils import _deconv_same_transpose_padding, _deconv_valid_transpose_padding
from ngclearn.utils.conv_utils import conv2d, deconv2d, rot180

################################################################################
# filter update computation
def calc_dK(x, d_out, delta_shape, stride_size=1, out_size = 2, padding = "SAME"): ## non-JIT wrapper
    deX, deY = delta_shape
    if deX > 0:
        return _calc_dK_subset(x, d_out, delta_shape, stride_size=stride_size, out_size = out_size, padding = padding)
    return _calc_dK(x, d_out, stride_size=stride_size, out_size = out_size, padding=padding)

@partial(jit, static_argnums=[2,3,4, 5])
def _calc_dK_subset(x, d_out, delta_shape, stride_size=1, out_size = 2, padding = "SAME"):
    deX, deY = delta_shape
    ## apply a pre-computation trimming step ("negative padding")
    _x = x[:,0:x.shape[1]-deX,0:x.shape[2]-deY,:]
    return _calc_dK(_x, d_out, stride_size=stride_size, out_size = out_size)

@partial(jit, static_argnums=[2,3,4])
def _calc_dK(x, d_out, stride_size=1, out_size = 2, padding = "SAME"):
    xT = jnp.transpose(x, axes=[3,1,2,0])
    d_out_T = jnp.transpose(d_out, axes=[1,2,0,3])
    if padding == "VALID":
        pad_args = _deconv_valid_transpose_padding(xT.shape[1], out_size, d_out_T.shape[1], stride_size)
    elif padding == "SAME":
        pad_args = _deconv_same_transpose_padding(xT.shape[1], out_size, d_out_T.shape[1], stride_size)
    dW = deconv2d(inputs=xT,
                filters=d_out_T,
                stride_size=stride_size,
                padding=pad_args)
    dW = jnp.transpose(dW, axes=[1,2,0,3])
    return dW
################################################################################
# input update computation
def calc_dX(K, d_out, delta_shape, stride_size=1, padding = ((0,0),(0,0))): ## non-JIT wrapper
    deX, deY = delta_shape
    if abs(deX) > 0 and stride_size > 1:
        return _calc_dX_subset(K, d_out, (abs(deX),abs(deY)), stride_size=stride_size, padding = padding)
    return _calc_dX(K, d_out, stride_size=stride_size, padding = padding)

@partial(jit, static_argnums=[2,3,4])
def _calc_dX_subset(K, d_out, delta_shape, stride_size=1, padding = ((0,0),(0,0))):
    deX, deY = delta_shape
    dx = _calc_dX(K, d_out, stride_size=stride_size, padding=padding)
    return dx

@partial(jit, static_argnums=[2,3])
def _calc_dX(K, d_out, stride_size=1, padding = ((0,0),(0,0))):
    ## deconvolution is done to get "through" a convolution backwards
    w_size = K.shape[0]
    K_T = rot180(K) #jnp.transpose(K, axes=[1,0,3,2])
    _pad = w_size - 1
    dx = conv2d(d_out,
                filters=K_T,
                stride_size=stride_size,
                padding = padding)
    return dx
################################################################################

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
