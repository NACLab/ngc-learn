from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
import ngclearn.utils.weight_distribution as dist
from ngclearn.components.synapses.convolution.ngcconv import conv2d
from ngcsimlib.logger import info
from ngclearn.utils import tensorstats

class ConvSynapse(JaxComponent): ## base-level convolutional cable
    """
    A base convolutional synaptic cable.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output
    | filters - current value tensor of filter/kernel efficacies
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

    # Define Functions
    def __init__(self, name, shape, x_shape, filter_init=None, bias_init=None, stride=1,
                 padding=None, resist_scale=1., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.filter_init = filter_init
        self.bias_init = bias_init

        ## Synapse meta-parameters
        self.shape = shape ## shape of synaptic filter tensor
        x_size, x_size = x_shape
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

        ######################### set up compartments ##########################
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
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias "
                            "kernel provided)!")
        self.biases = Compartment(dist.initialize_params(subkeys[2], bias_init,
                                                         (1, shape[1]))
                                  if bias_init else 0.0)

    @staticmethod
    def _advance_state(Rscale, padding, stride, weights, biases, inputs):
        _x = inputs
        outputs = conv2d(_x, weights, stride_size=stride, padding=padding) * Rscale + biases
        return outputs

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
            "synapse type": "ConvSynapse - performs a synaptic convolution (@) of inputs "
                         "to produce output signals"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX RNG key"},
            "parameter_compartments":
                {"filters": "Synaptic filter parameter values",
                 "biases": "Base-rate/bias parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic filter value matrix; `kernel width` x `kernel height` "
                     "x `number input channels` x `number output channels`",
            "x_shape": "Shape of any single incoming/input feature map",
            "weight_init": "Initialization conditions for synaptic filter (K) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level output scaling factor (R)",
            "stride": "length / size of stride",
            "padding": "pre-operator padding to use, i.e., `VALID` `SAME`"
        }
        info = {self.name: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @ inputs] * R + b",
                "hyperparameters": hyperparams}
        return info

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines
