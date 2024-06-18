from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
import ngclearn.utils.weight_distribution as dist
from ngcsimlib.logger import info
from ngclearn.utils.conv_utils import _conv_same_transpose_padding, _conv_valid_transpose_padding
from ngclearn.utils.conv_utils import *

@partial(jit, static_argnums=[2,3,4])
def calc_dK(x, d_out, delta_shape, stride_size=1, padding=((0, 0), (0, 0))):
    _x = x
    deX, deY = delta_shape
    if deX > 0:
        ## apply a pre-computation trimming step ("negative padding")
        _x = x[:, 0:x.shape[1]-deX, 0:x.shape[2]-deY, :]
    return _calc_dK(_x, d_out, stride_size=stride_size, padding=padding)

@partial(jit, static_argnums=[2, 3])
def _calc_dK(x, d_out, stride_size=1, padding=((0, 0), (0, 0))):
    xT = jnp.transpose(x, axes=[3, 1, 2, 0])
    d_out_T = jnp.transpose(d_out, axes=[1, 2, 0, 3])
    ## original conv2d
    dW = conv2d(inputs=xT,
                filters=d_out_T,
                stride_size=1,
                padding=padding,
                rhs_dilation=(stride_size,stride_size)).astype(jnp.float32)
    return jnp.transpose(dW, axes=[1,2,0,3])

################################################################################
# input update computation
def calc_dX(K, d_out, delta_shape, stride_size=1, anti_padding=None): ## non-JIT wrapper
    deX, deY = delta_shape
    if abs(deX) > 0 and stride_size > 1:
        return _calc_dX_subset(K, d_out, (abs(deX),abs(deY)), stride_size=stride_size,
                             anti_padding=anti_padding)
    else:
        return _calc_dX(K, d_out, stride_size=stride_size, anti_padding=anti_padding)

@partial(jit, static_argnums=[2,3,4])
def _calc_dX_subset(K, d_out, delta_shape, stride_size=1, anti_padding=None):
    deX, deY = delta_shape
    dx = _calc_dX(K, d_out, stride_size=stride_size, anti_padding=anti_padding)
    return dx

@partial(jit, static_argnums=[2,3])
def _calc_dX(K, d_out, stride_size=1, anti_padding=None):
    w_size = K.shape[0]
    K_T = rot180(K)  # Assuming rot180 is defined elsewhere.
    _pad = w_size - 1
    return deconv2d(d_out,
                  filters=K_T,
                  stride_size=stride_size,
                  padding=anti_padding).astype(jnp.float32)

class ConvSynapse(JaxComponent): ## static non-learnable synaptic cable
    """
    A base convolutional synaptic cable.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output
    | weights - current value tensor of kernel efficacies
    | biases - current base-rate/bias efficacies

    Args:
        name: the string name of this cell

        x_size: dimension of input signal (assuming a square input)

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number `filter height x filter width x input channels x number output channels`);
            note that currently filters/kernels are assumed to be square
            (kernel.width = kernel.height)

        filter_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values

        stride: length/size of stride

        padding: pre-operator padding to use -- "VALID" (none), "SAME"

        resist_scale: aa fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((K @ in) * resist_scale) + b
            where `@` denotes convolution

        batch_size: batch size dimension of this component
    """

    # Define Functions
    def __init__(self, name, shape, x_size, filter_init=None, bias_init=None, stride=1,
                 padding=None, resist_scale=1., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.filter_init = filter_init
        self.bias_init = bias_init

        ## Synapse meta-parameters
        self.shape = shape ## shape of synaptic filter tensor
        self.x_size = x_size
        self.Rscale = resist_scale ## post-transformation scale factor
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
        weights = dist.initialize_params(subkeys[0], filter_init, shape) ## filter tensor
        self.batch_size = batch_size # 1
        ## Compartment setup and shape computation
        _x = jnp.zeros((self.batch_size, x_size, x_size, n_in_chan))
        _d = conv2d(_x, weights, stride_size=stride, padding=padding) * 0
        self.in_shape = _x.shape
        self.out_shape = _d.shape
        self.inputs = Compartment(jnp.zeros(self.in_shape))
        self.outputs = Compartment(jnp.zeros(self.out_shape))
        self.weights = Compartment(weights)
        self.dWeights = Compartment(weights * 0)
        self.dInputs = Compartment(jnp.zeros(self.in_shape))
        self.pre = Compartment(jnp.zeros(self.in_shape))
        self.post = Compartment(jnp.zeros(self.out_shape))
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias "
                            "kernel provided)!")
        self.biases = Compartment(dist.initialize_params(subkeys[2], bias_init,
                                                         (1, shape[1]))
                                  if bias_init else 0.0)
        self.dBiases = Compartment(self.biases.value * 0)

        ########################################################################
        ## Shape error correction -- do shape correction inference (for local updates)
        self._init(self.batch_size, self.x_size, self.shape, self.stride,
                   self.padding, self.pad_args, self.weights)
        ########################################################################

    def _init(self, batch_size, x_size, shape, stride, padding, pad_args,
              weights):
        k_size, k_size, n_in_chan, n_out_chan = shape
        _x = jnp.zeros((batch_size, x_size, x_size, n_in_chan))
        _d = conv2d(_x, weights.value, stride_size=stride, padding=padding) * 0
        _dK = _calc_dK(_x, _d, stride_size=stride, padding=pad_args)
        ## get filter update correction
        dx = _dK.shape[0] - weights.value.shape[0]
        dy = _dK.shape[1] - weights.value.shape[1]
        self.delta_shape = (dx, dy)
        ## get input update correction
        _dx = _calc_dX(weights.value, _d, stride_size=stride,
                       anti_padding=pad_args)
        dx = (_dx.shape[1] - _x.shape[1])
        dy = (_dx.shape[2] - _x.shape[2])
        self.x_delta_shape = (dx, dy)

    @staticmethod
    def _advance_state(Rscale, padding, stride, weights, biases, inputs):
        _x = inputs
        outputs = conv2d(_x, weights, stride_size=stride, padding=padding) * Rscale + biases
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _evolve(bias_init, x_size, shape, stride, padding, pad_args, delta_shape,
                x_delta_shape, pre, post, weights): #, biases):
        k_size, k_size, n_in_chan, n_out_chan = shape
        ## calc dFilters
        dWeights = calc_dK(pre, post, delta_shape=delta_shape,
                           stride_size=stride, padding=pad_args)
        dBiases = 0. #jnp.zeros((1,1))
        if bias_init != None:
            dBiases = jnp.sum(post, axis=0, keepdims=True)
        ## calc dInputs
        antiPad = None
        if padding == "SAME":
            antiPad = _conv_same_transpose_padding(post.shape[1], x_size,
                                                   k_size, stride)
        elif padding == "VALID":
            antiPad = _conv_valid_transpose_padding(post.shape[1], x_size,
                                                    k_size, stride)
        dInputs = calc_dX(weights, post, delta_shape=x_delta_shape,
                          stride_size=stride, anti_padding=antiPad)
        return dWeights, dBiases, dInputs

    @resolver(_evolve)
    def evolve(self, dWeights, dBiases, dInputs):
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)
        self.dInputs.set(dInputs)

    @staticmethod
    def _reset(in_shape, out_shape):
        preVals = jnp.zeros(in_shape)
        postVals = jnp.zeros(out_shape)
        inputs = preVals
        outputs = postVals
        pre = preVals
        post = postVals
        return inputs, outputs, pre, post

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.bias_init != None:
            jnp.savez(file_name, weights=self.weights.value,
                      biases=self.biases.value)
        else:
            jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set(data['weights'])
        if "biases" in data.keys():
            self.biases.set(data['biases'])

    def help(self): ## component help function
        properties = {
            "cell type": "ConvSynapse - performs a synaptic convolution (@) of inputs "
                         "to produce output signals"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX RNG key"},
            "outputs_compartments":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic filter value matrix; `kernel width` x `kernel height` "
                     "x `number input channels` x `number output channels`",
            "weight_init": "Initialization conditions for synaptic filter (K) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level output scaling factor (R)"
        }
        info = {self.name: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @ inputs] * R + b",
                "hyperparameters": hyperparams}
        return info
